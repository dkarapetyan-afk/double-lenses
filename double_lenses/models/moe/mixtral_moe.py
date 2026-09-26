"""
Mixtral Sparse Mixture of Experts (SMoE) Lens.
Top-2 routing over 8 SwiGLU experts with exact forward and adjoint cotangent lifting.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from double_lenses.autodiff.tensor import Tensor, zeros
from double_lenses.autodiff.param_lens import LensContext, ParameterizedLens
from double_lenses.models.moe.router import MoERouterLens
from double_lenses.models.moe.expert import ExpertLens
from double_lenses.models.config import MixtralConfig


class MixtralMoELens(ParameterizedLens):
    """
    Mixtral Sparse Mixture of Experts (SMoE) Lens:
      - Router: Top-2 selection with softmax over num_experts
      - Experts: 8 independent SwiGLU feed-forward networks
      - Dispatch: Gathers tokens assigned to each expert
      - Combine: Sums outputs weighted by normalized gating weights
    """
    def __init__(self, name: str, config: MixtralConfig):
        super().__init__(name=name)
        self.config = config
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        # Router lens
        self.router = MoERouterLens(
            f"{name}.router",
            dim=self.dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            use_softmax=True,
        )
        for k, v in self.router.parameters.items():
            self.parameters[f"router.{k}"] = v

        # Expert lenses
        self.experts: List[ExpertLens] = []
        for i in range(self.num_experts):
            exp = ExpertLens(
                name=f"{name}.expert_{i}",
                in_dim=self.dim,
                hidden_dim=self.hidden_dim,
                out_dim=self.dim,
                expert_id=i,
            )
            self.experts.append(exp)
            for k, v in exp.parameters.items():
                self.parameters[f"expert_{i}.{k}"] = v

    def forward(self, ctx: LensContext, x: Tensor) -> Tensor:
        ctx_router = ctx.create_sub_context()
        routing_weights, selected_experts = self.router.forward(ctx_router, x)

        x_np = x.to_numpy()
        orig_shape = x_np.shape
        flat_x = x_np.reshape(-1, self.dim)
        n_tokens = flat_x.shape[0]

        flat_weights = routing_weights.reshape(-1, self.top_k)
        flat_experts = selected_experts.reshape(-1, self.top_k)

        flat_out = np.zeros_like(flat_x)

        # Cache expert inputs and contexts for adjoint
        expert_contexts: Dict[int, LensContext] = {}
        expert_token_indices: Dict[int, List[int]] = {i: [] for i in range(self.num_experts)}
        expert_k_slots: Dict[int, List[int]] = {i: [] for i in range(self.num_experts)}
        expert_outputs: Dict[int, np.ndarray] = {}

        # 1. Bucket tokens by assigned expert
        for t_idx in range(n_tokens):
            for k_idx in range(self.top_k):
                e_id = int(flat_experts[t_idx, k_idx])
                expert_token_indices[e_id].append(t_idx)
                expert_k_slots[e_id].append(k_idx)

        # 2. Run each expert on its gathered tokens
        for e_id in range(self.num_experts):
            t_indices = expert_token_indices[e_id]
            if not t_indices:
                continue

            sub_x = flat_x[t_indices]
            sub_ctx = ctx.create_sub_context()
            expert_contexts[e_id] = sub_ctx

            exp_out = self.experts[e_id].forward(sub_ctx, Tensor(sub_x, device=x.device)).to_numpy()
            expert_outputs[e_id] = exp_out

            k_slots = expert_k_slots[e_id]
            w = flat_weights[t_indices, k_slots, None]
            flat_out[t_indices] += exp_out * w

        out_np = flat_out.reshape(orig_shape)

        # Save for adjoint pass
        ctx.save("flat_x", flat_x)
        ctx.save("flat_weights", flat_weights)
        ctx.save("flat_experts", flat_experts)
        ctx.save("expert_contexts", expert_contexts)
        ctx.save("expert_token_indices", expert_token_indices)
        ctx.save("expert_k_slots", expert_k_slots)
        ctx.save("expert_outputs", expert_outputs)
        ctx.save("orig_shape", orig_shape)
        ctx.save("n_tokens", n_tokens)

        return Tensor(out_np, device=x.device)

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        ctx_router = ctx.sub_contexts[0]
        flat_x = ctx.get("flat_x")
        flat_weights = ctx.get("flat_weights")
        flat_experts = ctx.get("flat_experts")
        expert_contexts = ctx.get("expert_contexts")
        expert_token_indices = ctx.get("expert_token_indices")
        expert_k_slots = ctx.get("expert_k_slots")
        expert_outputs = ctx.get("expert_outputs")
        orig_shape = ctx.get("orig_shape")
        n_tokens = ctx.get("n_tokens")

        dy_np = grad_y.to_numpy().reshape(-1, self.dim)

        grad_flat_x = np.zeros_like(flat_x)
        grad_flat_weights = np.zeros_like(flat_weights)

        # 1. Pullback through expert paths and routing weights
        for e_id in range(self.num_experts):
            t_indices = expert_token_indices[e_id]
            if not t_indices:
                continue

            k_slots = expert_k_slots[e_id]
            w = flat_weights[t_indices, k_slots, None]
            dy_sub = dy_np[t_indices]
            exp_out = expert_outputs[e_id]

            # Gradient w.r.t routing weights: dL/dw = dy * exp_out
            dw = np.sum(dy_sub * exp_out, axis=-1)
            grad_flat_weights[t_indices, k_slots] += dw

            # Gradient w.r.t expert output: dL/d(exp_out) = dy * w
            d_exp_out = dy_sub * w
            sub_ctx = expert_contexts[e_id]

            grad_sub_x = self.experts[e_id].adjoint(sub_ctx, Tensor(d_exp_out, device=grad_y.device)).to_numpy()
            grad_flat_x[t_indices] += grad_sub_x

        # 2. Pullback through router
        grad_weights = grad_flat_weights.reshape(orig_shape[:-1] + (self.top_k,))
        grad_x_router = self.router.adjoint(ctx_router, grad_weights)

        # 3. Combine activation cotangents
        grad_x = Tensor(grad_flat_x.reshape(orig_shape), device=grad_y.device) + grad_x_router
        return grad_x
