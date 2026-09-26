"""
Grouped-Query Attention (GQA) Lens for Mixtral.
Composed entirely of categorical lenses with exact forward and adjoint passes.
"""

from typing import Optional, Tuple
import numpy as np
from double_lenses.autodiff.tensor import Tensor, randn, zeros
from double_lenses.autodiff.param_lens import LensContext, ParameterizedLens
from double_lenses.autodiff.layers import LinearLens, RoPELens, SoftmaxLens
from double_lenses.models.config import MixtralConfig


class GroupedQueryAttentionLens(ParameterizedLens):
    """
    Grouped-Query Attention (GQA) Lens used in Mixtral.
    Combines Q, K, V projections, RoPE, head repetition,
    scaled dot-product attention, and output projection.
    """
    def __init__(self, name: str, config: MixtralConfig):
        super().__init__(name=name)
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_rep = self.n_heads // self.n_kv_heads

        # Projections
        self.q_proj = LinearLens(f"{name}.q_proj", self.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = LinearLens(f"{name}.k_proj", self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = LinearLens(f"{name}.v_proj", self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = LinearLens(f"{name}.out_proj", self.n_heads * self.head_dim, self.dim, bias=False)

        # RoPE
        self.rope = RoPELens(f"{name}.rope", self.head_dim, config.max_seq_len, config.rope_theta)

        # Softmax
        self.softmax = SoftmaxLens(f"{name}.attn_softmax", axis=-1)

        # Register parameters
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            for k, v in proj.parameters.items():
                self.parameters[f"{proj.name}.{k}"] = v

    def forward(self, ctx: LensContext, x: Tensor, start_pos: int = 0) -> Tensor:
        ctx_q = ctx.create_sub_context()
        ctx_k = ctx.create_sub_context()
        ctx_v = ctx.create_sub_context()
        ctx_rope_q = ctx.create_sub_context()
        ctx_rope_k = ctx.create_sub_context()
        ctx_sm = ctx.create_sub_context()
        ctx_out = ctx.create_sub_context()

        batch_size, seq_len, _ = x.shape

        # 1. Projections
        q = self.q_proj.forward(ctx_q, x)  # (batch, seq, n_heads * head_dim)
        k = self.k_proj.forward(ctx_k, x)  # (batch, seq, n_kv_heads * head_dim)
        v = self.v_proj.forward(ctx_v, x)  # (batch, seq, n_kv_heads * head_dim)

        # 2. Reshape to heads: (batch, seq, n_heads, head_dim)
        q_heads = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k_heads = k.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v_heads = v.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # 3. Apply RoPE
        q_rope = self.rope.forward(ctx_rope_q, q_heads, start_pos=start_pos)
        k_rope = self.rope.forward(ctx_rope_k, k_heads, start_pos=start_pos)

        # 4. Transpose for attention: (batch, n_heads, seq_len, head_dim)
        # and repeat KV heads if n_kv_heads < n_heads
        q_np = q_rope.to_numpy().transpose(0, 2, 1, 3)
        k_np = k_rope.to_numpy().transpose(0, 2, 1, 3)
        v_np = v_heads.to_numpy().transpose(0, 2, 1, 3)

        if self.num_rep > 1:
            k_np = np.repeat(k_np, self.num_rep, axis=1)
            v_np = np.repeat(v_np, self.num_rep, axis=1)

        # 5. Scaled dot-product attention
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.matmul(q_np, k_np.transpose(0, 1, 3, 2)) * scale

        # Causal mask (if seq_len > 1)
        if seq_len > 1:
            mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
            scores = scores + mask[None, None, :, :]

        attn_weights = self.softmax.forward(ctx_sm, Tensor(scores, device=x.device)).to_numpy()

        # Output context
        context = np.matmul(attn_weights, v_np)  # (batch, n_heads, seq_len, head_dim)
        context_trans = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.n_heads * self.head_dim)

        out = self.out_proj.forward(ctx_out, Tensor(context_trans, device=x.device))

        # Save intermediates for adjoint pass
        ctx.save("q_np", q_np)
        ctx.save("k_np", k_np)
        ctx.save("v_np", v_np)
        ctx.save("attn_weights", attn_weights)
        ctx.save("batch_size", batch_size)
        ctx.save("seq_len", seq_len)

        return out

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        ctx_q, ctx_k, ctx_v, ctx_rope_q, ctx_rope_k, ctx_sm, ctx_out = ctx.sub_contexts[:7]
        q_np = ctx.get("q_np")
        k_np = ctx.get("k_np")
        v_np = ctx.get("v_np")
        attn_weights = ctx.get("attn_weights")
        batch_size = ctx.get("batch_size")
        seq_len = ctx.get("seq_len")

        # 1. Pullback through out_proj
        grad_context_trans = self.out_proj.adjoint(ctx_out, grad_y)
        grad_context = grad_context_trans.to_numpy().reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # 2. Pullback through context = attn_weights @ v_np
        # grad_v_np = attn_weights^T @ grad_context
        grad_v_np = np.matmul(attn_weights.transpose(0, 1, 3, 2), grad_context)
        # grad_attn_weights = grad_context @ v_np^T
        grad_attn_weights = np.matmul(grad_context, v_np.transpose(0, 1, 3, 2))

        # 3. Pullback through softmax
        grad_scores = self.softmax.adjoint(ctx_sm, Tensor(grad_attn_weights, device=grad_y.device)).to_numpy()

        # 4. Pullback through scaled dot product
        scale = 1.0 / np.sqrt(self.head_dim)
        # scores = (q_np @ k_np^T) * scale
        grad_q_np = np.matmul(grad_scores, k_np) * scale
        grad_k_np = np.matmul(grad_scores.transpose(0, 1, 3, 2), q_np) * scale

        # 5. Handle num_rep for KV
        if self.num_rep > 1:
            # Sum gradients across repeated heads
            grad_k_np = grad_k_np.reshape(batch_size, self.n_kv_heads, self.num_rep, seq_len, self.head_dim).sum(axis=2)
            grad_v_np = grad_v_np.reshape(batch_size, self.n_kv_heads, self.num_rep, seq_len, self.head_dim).sum(axis=2)

        # 6. Transpose back: (batch, seq_len, n_heads, head_dim)
        grad_q_heads = Tensor(grad_q_np.transpose(0, 2, 1, 3), device=grad_y.device)
        grad_k_heads = Tensor(grad_k_np.transpose(0, 2, 1, 3), device=grad_y.device)
        grad_v_heads = Tensor(grad_v_np.transpose(0, 2, 1, 3), device=grad_y.device)

        # 7. Pullback through RoPE
        grad_q_unrope = self.rope.adjoint(ctx_rope_q, grad_q_heads)
        grad_k_unrope = self.rope.adjoint(ctx_rope_k, grad_k_heads)

        # 8. Reshape to flat projection outputs
        grad_q = grad_q_unrope.reshape(batch_size, seq_len, self.n_heads * self.head_dim)
        grad_k = grad_k_unrope.reshape(batch_size, seq_len, self.n_kv_heads * self.head_dim)
        grad_v = grad_v_heads.reshape(batch_size, seq_len, self.n_kv_heads * self.head_dim)

        # 9. Pullback through linear projections
        grad_x_q = self.q_proj.adjoint(ctx_q, grad_q)
        grad_x_k = self.k_proj.adjoint(ctx_k, grad_k)
        grad_x_v = self.v_proj.adjoint(ctx_v, grad_v)

        # 10. Combine input cotangent
        grad_x = grad_x_q + grad_x_k + grad_x_v
        return grad_x
