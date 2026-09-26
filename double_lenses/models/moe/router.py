"""
Gating Router Lens for Mixture of Experts (MoE).
Supports Top-K selection with Softmax (Mixtral) or Sigmoid (DeepSeek) normalization
and exact analytical adjoint cotangent pullback.
"""

from typing import Optional, Tuple
import numpy as np
from double_lenses.autodiff.tensor import Tensor, randn, zeros
from double_lenses.autodiff.param_lens import LensContext, ParameterizedLens
from double_lenses.autodiff.layers import LinearLens


class MoERouterLens(ParameterizedLens):
    """
    MoE Router Lens:
      - Projects tokens: logits = x W_gate^T
      - Selects top-k experts per token
      - Normalizes routing weights via softmax or sigmoid
    """
    def __init__(self, name: str, dim: int, num_experts: int, top_k: int = 2, use_softmax: bool = True):
        super().__init__(name=name)
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_softmax = use_softmax

        self.w_gate = LinearLens(f"{name}.w_gate", dim, num_experts, bias=False)
        for k, v in self.w_gate.parameters.items():
            self.parameters[f"w_gate.{k}"] = v

    def forward(self, ctx: LensContext, x: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          routing_weights: (..., top_k)
          selected_experts: (..., top_k) int indices
        """
        ctx_gate = ctx.create_sub_context()
        logits = self.w_gate.forward(ctx_gate, x)
        logits_np = logits.to_numpy()  # (..., num_experts)
        orig_shape = logits_np.shape
        flat_logits = logits_np.reshape(-1, self.num_experts)
        n_tokens = flat_logits.shape[0]

        # Top-k selection
        top_indices = np.argsort(-flat_logits, axis=-1)[:, :self.top_k]  # (n_tokens, top_k)
        top_logits = np.take_along_axis(flat_logits, top_indices, axis=-1)  # (n_tokens, top_k)

        if self.use_softmax:
            max_val = np.max(top_logits, axis=-1, keepdims=True)
            exp_val = np.exp(top_logits - max_val)
            weights = exp_val / np.sum(exp_val, axis=-1, keepdims=True)
        else:
            # Sigmoid routing (DeepSeek style)
            sig = 1.0 / (1.0 + np.exp(-np.clip(top_logits, -50.0, 50.0)))
            weights = sig / np.sum(sig, axis=-1, keepdims=True)

        routing_weights = weights.reshape(orig_shape[:-1] + (self.top_k,))
        selected_experts = top_indices.reshape(orig_shape[:-1] + (self.top_k,))

        ctx.save("flat_logits", flat_logits)
        ctx.save("top_indices", top_indices)
        ctx.save("weights", weights)
        ctx.save("orig_shape", orig_shape)
        ctx.save("n_tokens", n_tokens)

        return routing_weights, selected_experts

    def adjoint(self, ctx: LensContext, grad_weights: np.ndarray) -> Tensor:
        """
        grad_weights: cotangents w.r.t routing_weights, shape (..., top_k)
        Returns:
          grad_x: cotangent w.r.t input tokens x
        """
        ctx_gate = ctx.sub_contexts[0]
        top_indices = ctx.get("top_indices")
        weights = ctx.get("weights")
        orig_shape = ctx.get("orig_shape")
        n_tokens = ctx.get("n_tokens")

        flat_gw = grad_weights.reshape(-1, self.top_k)

        # Pullback through normalization
        if self.use_softmax:
            # J_softmax^T @ grad_weights
            sum_w_gw = np.sum(weights * flat_gw, axis=-1, keepdims=True)
            d_top_logits = weights * (flat_gw - sum_w_gw)
        else:
            sum_w_gw = np.sum(weights * flat_gw, axis=-1, keepdims=True)
            d_norm = weights * (flat_gw - sum_w_gw)
            d_top_logits = d_norm * (1.0 - weights)

        # Scatter d_top_logits back to (n_tokens, num_experts)
        grad_flat_logits = np.zeros((n_tokens, self.num_experts), dtype=np.float32)
        np.put_along_axis(grad_flat_logits, top_indices, d_top_logits, axis=-1)

        grad_logits = Tensor(grad_flat_logits.reshape(orig_shape), device=self.w_gate.w.device)

        # Pullback through w_gate
        grad_x = self.w_gate.adjoint(ctx_gate, grad_logits)
        return grad_x
