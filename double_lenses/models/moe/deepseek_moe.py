"""
DeepSeekMoE Lens Architecture: Fine-Grained Routed Experts + Isolated Shared Experts.
Implements exact forward and analytical adjoint lifting.
"""

import numpy as np

from double_lenses.autodiff.param_lens import LensContext, ParameterizedLens
from double_lenses.autodiff.tensor import Tensor
from double_lenses.models.config import DeepSeekConfig
from double_lenses.models.moe.expert import ExpertLens
from double_lenses.models.moe.router import MoERouterLens


class DeepSeekMoELens(ParameterizedLens):
    """
    DeepSeekMoE Architecture:
      - Shared Experts: m shared experts always activated for every token.
      - Routed Experts: N fine-grained routed experts, top-k routed per token.
      - Output: y = Σ shared_m(x) + Σ w_k * routed_k(x).
    """

    def __init__(self, name: str, config: DeepSeekConfig):
        super().__init__(name=name)
        self.config = config
        self.dim = config.dim
        self.num_routed = config.num_routed_experts
        self.num_shared = config.num_shared_experts
        self.top_k = config.top_k
        self.routed_hidden_dim = config.routed_hidden_dim
        self.shared_hidden_dim = config.shared_hidden_dim

        # Router for routed experts
        self.router = MoERouterLens(
            f"{name}.router",
            dim=self.dim,
            num_experts=self.num_routed,
            top_k=self.top_k,
            use_softmax=False,  # Sigmoid-based routing as in DeepSeek-V2/V3
        )
        for k, v in self.router.parameters.items():
            self.parameters[f"router.{k}"] = v

        # Shared Experts
        self.shared_experts: list[ExpertLens] = []
        for i in range(self.num_shared):
            exp = ExpertLens(
                name=f"{name}.shared_{i}",
                in_dim=self.dim,
                hidden_dim=self.shared_hidden_dim,
                out_dim=self.dim,
                expert_id=i,
            )
            self.shared_experts.append(exp)
            for k, v in exp.parameters.items():
                self.parameters[f"shared_{i}.{k}"] = v

        # Routed Experts
        self.routed_experts: list[ExpertLens] = []
        for i in range(self.num_routed):
            exp = ExpertLens(
                name=f"{name}.routed_{i}",
                in_dim=self.dim,
                hidden_dim=self.routed_hidden_dim,
                out_dim=self.dim,
                expert_id=i,
            )
            self.routed_experts.append(exp)
            for k, v in exp.parameters.items():
                self.parameters[f"routed_{i}.{k}"] = v

    def forward(self, ctx: LensContext, x: Tensor) -> Tensor:
        ctx_router = ctx.create_sub_context()
        routing_weights, selected_experts = self.router.forward(ctx_router, x)

        x_np = x.to_numpy()
        orig_shape = x_np.shape
        flat_x = x_np.reshape(-1, self.dim)
        n_tokens = flat_x.shape[0]

        flat_weights = routing_weights.reshape(-1, self.top_k)
        flat_experts = selected_experts.reshape(-1, self.top_k)

        # 1. Run Shared Experts (executed on ALL tokens)
        shared_out = np.zeros_like(flat_x)
        shared_contexts: list[LensContext] = []
        for _i, s_exp in enumerate(self.shared_experts):
            s_ctx = ctx.create_sub_context()
            shared_contexts.append(s_ctx)
            s_res = s_exp.forward(s_ctx, Tensor(flat_x, device=x.device)).to_numpy()
            shared_out += s_res

        # 2. Run Routed Experts (executed on gathered tokens)
        routed_out = np.zeros_like(flat_x)
        expert_contexts: dict[int, LensContext] = {}
        expert_token_indices: dict[int, list[int]] = {i: [] for i in range(self.num_routed)}
        expert_k_slots: dict[int, list[int]] = {i: [] for i in range(self.num_routed)}
        expert_outputs: dict[int, np.ndarray] = {}

        for t_idx in range(n_tokens):
            for k_idx in range(self.top_k):
                e_id = int(flat_experts[t_idx, k_idx])
                expert_token_indices[e_id].append(t_idx)
                expert_k_slots[e_id].append(k_idx)

        for e_id in range(self.num_routed):
            t_indices = expert_token_indices[e_id]
            if not t_indices:
                continue

            sub_x = flat_x[t_indices]
            sub_ctx = ctx.create_sub_context()
            expert_contexts[e_id] = sub_ctx

            exp_out = self.routed_experts[e_id].forward(sub_ctx, Tensor(sub_x, device=x.device)).to_numpy()
            expert_outputs[e_id] = exp_out

            k_slots = expert_k_slots[e_id]
            w = flat_weights[t_indices, k_slots, None]
            routed_out[t_indices] += exp_out * w

        # Total output is shared + routed
        total_out = (shared_out + routed_out).reshape(orig_shape)

        # Save for adjoint
        ctx.save("flat_x", flat_x)
        ctx.save("flat_weights", flat_weights)
        ctx.save("flat_experts", flat_experts)
        ctx.save("shared_contexts", shared_contexts)
        ctx.save("expert_contexts", expert_contexts)
        ctx.save("expert_token_indices", expert_token_indices)
        ctx.save("expert_k_slots", expert_k_slots)
        ctx.save("expert_outputs", expert_outputs)
        ctx.save("orig_shape", orig_shape)
        ctx.save("n_tokens", n_tokens)

        return Tensor(total_out, device=x.device)

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        ctx_router = ctx.sub_contexts[0]
        flat_x = ctx.get("flat_x")
        flat_weights = ctx.get("flat_weights")
        shared_contexts = ctx.get("shared_contexts")
        expert_contexts = ctx.get("expert_contexts")
        expert_token_indices = ctx.get("expert_token_indices")
        expert_k_slots = ctx.get("expert_k_slots")
        expert_outputs = ctx.get("expert_outputs")
        orig_shape = ctx.get("orig_shape")

        dy_np = grad_y.to_numpy().reshape(-1, self.dim)

        grad_flat_x = np.zeros_like(flat_x)
        grad_flat_weights = np.zeros_like(flat_weights)

        # 1. Pullback through shared experts
        for i, s_exp in enumerate(self.shared_experts):
            s_ctx = shared_contexts[i]
            grad_s = s_exp.adjoint(s_ctx, Tensor(dy_np, device=grad_y.device)).to_numpy()
            grad_flat_x += grad_s

        # 2. Pullback through routed experts
        for e_id in range(self.num_routed):
            t_indices = expert_token_indices[e_id]
            if not t_indices:
                continue

            k_slots = expert_k_slots[e_id]
            w = flat_weights[t_indices, k_slots, None]
            dy_sub = dy_np[t_indices]
            exp_out = expert_outputs[e_id]

            dw = np.sum(dy_sub * exp_out, axis=-1)
            grad_flat_weights[t_indices, k_slots] += dw

            d_exp_out = dy_sub * w
            sub_ctx = expert_contexts[e_id]
            grad_sub_x = self.routed_experts[e_id].adjoint(sub_ctx, Tensor(d_exp_out, device=grad_y.device)).to_numpy()
            grad_flat_x[t_indices] += grad_sub_x

        # 3. Pullback through router
        grad_weights = grad_flat_weights.reshape(orig_shape[:-1] + (self.top_k,))
        grad_x_router = self.router.adjoint(ctx_router, grad_weights)

        # 4. Total cotangent
        grad_x = Tensor(grad_flat_x.reshape(orig_shape), device=grad_y.device) + grad_x_router
        return grad_x
