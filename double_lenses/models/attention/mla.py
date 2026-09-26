"""
Multi-Head Latent Attention (MLA) Lens for DeepSeek-V2/V3.
Implements low-rank KV compression, decoupled RoPE, and exact analytical adjoint lifting.
"""

import numpy as np

from double_lenses.autodiff.layers import LinearLens, RoPELens, SoftmaxLens
from double_lenses.autodiff.param_lens import LensContext, ParameterizedLens
from double_lenses.autodiff.tensor import Tensor
from double_lenses.models.config import DeepSeekConfig


class MultiHeadLatentAttentionLens(ParameterizedLens):
    """
    DeepSeek Multi-Head Latent Attention (MLA) Lens:
      - Compresses Keys and Values into a low-rank latent vector c_t^{KV} = x_t W_{DKV}
      - Decompresses content keys k^C = c^{KV} W_{UK} and values v^C = c^{KV} W_{UV}
      - Decoupled RoPE keys k^R = RoPE(x_t W_{KR})
      - Compresses Queries c_t^Q = x_t W_{DQ}, decompresses q^C = c^Q W_{UQ}, q^R = RoPE(c^Q W_{QR})
      - Attention score combines content matching with decoupled positional matching:
          Score = (q^C k^{C,T} + q^R k^{R,T}) / sqrt(d_h + d_R)
    """

    def __init__(self, name: str, config: DeepSeekConfig):
        super().__init__(name=name)
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.head_dim = config.v_head_dim

        # Projections for KV
        self.w_dkv = LinearLens(f"{name}.w_dkv", self.dim, self.kv_lora_rank, bias=False)
        self.w_uk = LinearLens(f"{name}.w_uk", self.kv_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.w_uv = LinearLens(f"{name}.w_uv", self.kv_lora_rank, self.n_heads * self.v_head_dim, bias=False)
        self.w_kr = LinearLens(f"{name}.w_kr", self.dim, self.qk_rope_head_dim, bias=False)

        # Projections for Q
        self.w_dq = LinearLens(f"{name}.w_dq", self.dim, self.q_lora_rank, bias=False)
        self.w_uq = LinearLens(f"{name}.w_uq", self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.w_qr = LinearLens(f"{name}.w_qr", self.q_lora_rank, self.n_heads * self.qk_rope_head_dim, bias=False)

        # Output projection
        self.out_proj = LinearLens(f"{name}.out_proj", self.n_heads * self.v_head_dim, self.dim, bias=False)

        # RoPE lenses
        self.rope_k = RoPELens(f"{name}.rope_k", self.qk_rope_head_dim, config.max_seq_len, config.rope_theta)
        self.rope_q = RoPELens(f"{name}.rope_q", self.qk_rope_head_dim, config.max_seq_len, config.rope_theta)

        # Softmax
        self.softmax = SoftmaxLens(f"{name}.attn_softmax", axis=-1)

        # Register parameters
        projs = [self.w_dkv, self.w_uk, self.w_uv, self.w_kr, self.w_dq, self.w_uq, self.w_qr, self.out_proj]
        for p in projs:
            for k, v in p.parameters.items():
                self.parameters[f"{p.name}.{k}"] = v

    def forward(self, ctx: LensContext, x: Tensor, start_pos: int = 0) -> Tensor:
        ctx_dkv = ctx.create_sub_context()
        ctx_uk = ctx.create_sub_context()
        ctx_uv = ctx.create_sub_context()
        ctx_kr = ctx.create_sub_context()
        ctx_dq = ctx.create_sub_context()
        ctx_uq = ctx.create_sub_context()
        ctx_qr = ctx.create_sub_context()
        ctx_rope_k = ctx.create_sub_context()
        ctx_rope_q = ctx.create_sub_context()
        ctx_sm = ctx.create_sub_context()
        ctx_out = ctx.create_sub_context()

        batch_size, seq_len, _ = x.shape

        # 1. KV Path: Compress to latent space c_kv
        c_kv = self.w_dkv.forward(ctx_dkv, x)  # (batch, seq, kv_lora_rank)
        k_c = self.w_uk.forward(ctx_uk, c_kv).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v_c = self.w_uv.forward(ctx_uv, c_kv).reshape(batch_size, seq_len, self.n_heads, self.v_head_dim)

        # Decoupled RoPE Key
        k_r_raw = self.w_kr.forward(ctx_kr, x).reshape(batch_size, seq_len, 1, self.qk_rope_head_dim)
        k_r = self.rope_k.forward(ctx_rope_k, k_r_raw, start_pos=start_pos)

        # 2. Q Path: Compress to latent space c_q
        c_q = self.w_dq.forward(ctx_dq, x)  # (batch, seq, q_lora_rank)
        q_c = self.w_uq.forward(ctx_uq, c_q).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        q_r_raw = self.w_qr.forward(ctx_qr, c_q).reshape(batch_size, seq_len, self.n_heads, self.qk_rope_head_dim)
        q_r = self.rope_q.forward(ctx_rope_q, q_r_raw, start_pos=start_pos)

        # 3. Transpose for Attention: (batch, n_heads, seq_len, -1)
        qc_np = q_c.to_numpy().transpose(0, 2, 1, 3)
        kc_np = k_c.to_numpy().transpose(0, 2, 1, 3)
        vc_np = v_c.to_numpy().transpose(0, 2, 1, 3)

        qr_np = q_r.to_numpy().transpose(0, 2, 1, 3)
        # Broadcast k_r along n_heads
        kr_np = np.repeat(k_r.to_numpy().transpose(0, 2, 1, 3), self.n_heads, axis=1)

        # 4. Attention Score = (q_c @ k_c^T + q_r @ k_r^T) / sqrt(head_dim + qk_rope_head_dim)
        scale = 1.0 / np.sqrt(self.head_dim + self.qk_rope_head_dim)
        scores_c = np.matmul(qc_np, kc_np.transpose(0, 1, 3, 2))
        scores_r = np.matmul(qr_np, kr_np.transpose(0, 1, 3, 2))
        scores = (scores_c + scores_r) * scale

        if seq_len > 1:
            mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
            scores = scores + mask[None, None, :, :]

        attn_weights = self.softmax.forward(ctx_sm, Tensor(scores, device=x.device)).to_numpy()

        # 5. Output
        context = np.matmul(attn_weights, vc_np)  # (batch, n_heads, seq, v_head_dim)
        context_trans = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.n_heads * self.v_head_dim)

        out = self.out_proj.forward(ctx_out, Tensor(context_trans, device=x.device))

        # Save for adjoint
        ctx.save("qc_np", qc_np)
        ctx.save("kc_np", kc_np)
        ctx.save("vc_np", vc_np)
        ctx.save("qr_np", qr_np)
        ctx.save("kr_np", kr_np)
        ctx.save("attn_weights", attn_weights)
        ctx.save("batch_size", batch_size)
        ctx.save("seq_len", seq_len)

        return out

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        (ctx_dkv, ctx_uk, ctx_uv, ctx_kr, ctx_dq, ctx_uq, ctx_qr, ctx_rope_k, ctx_rope_q, ctx_sm, ctx_out) = (
            ctx.sub_contexts[:11]
        )

        qc_np = ctx.get("qc_np")
        kc_np = ctx.get("kc_np")
        vc_np = ctx.get("vc_np")
        qr_np = ctx.get("qr_np")
        kr_np = ctx.get("kr_np")
        attn_weights = ctx.get("attn_weights")
        batch_size = ctx.get("batch_size")
        seq_len = ctx.get("seq_len")

        # 1. Pullback through out_proj
        grad_context_trans = self.out_proj.adjoint(ctx_out, grad_y)
        grad_context = (
            grad_context_trans.to_numpy()
            .reshape(batch_size, seq_len, self.n_heads, self.v_head_dim)
            .transpose(0, 2, 1, 3)
        )

        # 2. Pullback through context = attn_weights @ vc_np
        grad_vc_np = np.matmul(attn_weights.transpose(0, 1, 3, 2), grad_context)
        grad_attn_weights = np.matmul(grad_context, vc_np.transpose(0, 1, 3, 2))

        # 3. Pullback through softmax
        grad_scores = self.softmax.adjoint(ctx_sm, Tensor(grad_attn_weights, device=grad_y.device)).to_numpy()

        # 4. Pullback through scaled scores
        scale = 1.0 / np.sqrt(self.head_dim + self.qk_rope_head_dim)
        grad_scores_scaled = grad_scores * scale

        # Content query/key gradients
        grad_qc_np = np.matmul(grad_scores_scaled, kc_np)
        grad_kc_np = np.matmul(grad_scores_scaled.transpose(0, 1, 3, 2), qc_np)

        # RoPE query/key gradients
        grad_qr_np = np.matmul(grad_scores_scaled, kr_np)
        grad_kr_np = np.matmul(grad_scores_scaled.transpose(0, 1, 3, 2), qr_np)
        # Sum kr across n_heads broadcast
        grad_kr_sum = grad_kr_np.sum(axis=1, keepdims=True)

        # 5. Reshape and pull back RoPE
        grad_qr = self.rope_q.adjoint(
            ctx_rope_q, Tensor(grad_qr_np.transpose(0, 2, 1, 3), device=grad_y.device)
        ).reshape(batch_size, seq_len, self.n_heads * self.qk_rope_head_dim)

        grad_kr = self.rope_k.adjoint(
            ctx_rope_k, Tensor(grad_kr_sum.transpose(0, 2, 1, 3), device=grad_y.device)
        ).reshape(batch_size, seq_len, self.qk_rope_head_dim)

        grad_qc = Tensor(
            grad_qc_np.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.n_heads * self.head_dim),
            device=grad_y.device,
        )
        grad_kc = Tensor(
            grad_kc_np.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.n_heads * self.head_dim),
            device=grad_y.device,
        )
        grad_vc = Tensor(
            grad_vc_np.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.n_heads * self.v_head_dim),
            device=grad_y.device,
        )

        # 6. Q Path Pullback: c_q
        grad_cq_uq = self.w_uq.adjoint(ctx_uq, grad_qc)
        grad_cq_qr = self.w_qr.adjoint(ctx_qr, grad_qr)
        grad_cq = grad_cq_uq + grad_cq_qr
        grad_x_q = self.w_dq.adjoint(ctx_dq, grad_cq)

        # 7. KV Path Pullback: c_kv
        grad_ckv_uk = self.w_uk.adjoint(ctx_uk, grad_kc)
        grad_ckv_uv = self.w_uv.adjoint(ctx_uv, grad_vc)
        grad_ckv = grad_ckv_uk + grad_ckv_uv
        grad_x_kv = self.w_dkv.adjoint(ctx_dkv, grad_ckv)

        # 8. Decoupled Key RoPE Pullback
        grad_x_kr = self.w_kr.adjoint(ctx_kr, grad_kr)

        # 9. Aggregate input cotangents
        grad_x = grad_x_q + grad_x_kv + grad_x_kr
        return grad_x
