"""
Categorical Loss Lenses with Exact Cotangent Pullback Generation.
Provides terminal lenses for computing scalar loss and backward seed cotangents.
"""

from typing import Any, Tuple, Union
import numpy as np
from double_lenses.autodiff.tensor import Tensor
from double_lenses.autodiff.param_lens import LensContext, ParameterizedLens


class CrossEntropyLossLens(ParameterizedLens):
    """
    Cross Entropy Loss over logits and integer target labels.
    Forward: Computes scalar mean loss.
    Adjoint: Generates initial seed cotangent:
      ȳ = (softmax(logits) - one_hot(targets)) / N
    """
    def __init__(self, name: str = "cross_entropy"):
        super().__init__(name=name)

    def forward(self, ctx: LensContext, logits: Tensor, targets: Union[Tensor, np.ndarray]) -> Tensor:
        l_np = logits.to_numpy()
        t_np = targets.to_numpy() if isinstance(targets, Tensor) else np.array(targets)

        # Numerically stable log-softmax
        max_l = np.max(l_np, axis=-1, keepdims=True)
        exp_l = np.exp(l_np - max_l)
        sum_exp = np.sum(exp_l, axis=-1, keepdims=True)
        log_probs = (l_np - max_l) - np.log(sum_exp)
        probs = exp_l / sum_exp

        # Flatten for loss indexing
        flat_lp = log_probs.reshape(-1, log_probs.shape[-1])
        flat_t = t_np.reshape(-1)
        n_samples = flat_t.shape[0]

        loss_val = -np.mean(flat_lp[np.arange(n_samples), flat_t])

        ctx.save("probs", probs)
        ctx.save("targets", t_np)
        ctx.save("n_samples", n_samples)

        return Tensor(np.array(loss_val, dtype=np.float32), device=logits.device)

    def adjoint(self, ctx: LensContext, grad_loss: Union[Tensor, float] = 1.0) -> Tensor:
        probs = ctx.get("probs")
        targets = ctx.get("targets")
        n_samples = ctx.get("n_samples")

        scale = grad_loss.item() if isinstance(grad_loss, Tensor) else float(grad_loss)

        grad_logits = probs.copy()
        flat_gl = grad_logits.reshape(-1, grad_logits.shape[-1])
        flat_t = targets.reshape(-1)

        flat_gl[np.arange(n_samples), flat_t] -= 1.0
        grad_logits = (grad_logits / n_samples) * scale

        return Tensor(grad_logits, device="cpu")


class MSELossLens(ParameterizedLens):
    """Mean Squared Error Loss."""
    def __init__(self, name: str = "mse"):
        super().__init__(name=name)

    def forward(self, ctx: LensContext, pred: Tensor, target: Tensor) -> Tensor:
        p_np = pred.to_numpy()
        t_np = target.to_numpy()
        diff = p_np - t_np
        loss_val = np.mean(diff ** 2)
        ctx.save("diff", diff)
        ctx.save("size", p_np.size)
        return Tensor(np.array(loss_val, dtype=np.float32), device=pred.device)

    def adjoint(self, ctx: LensContext, grad_loss: Union[Tensor, float] = 1.0) -> Tensor:
        diff = ctx.get("diff")
        size = ctx.get("size")
        scale = grad_loss.item() if isinstance(grad_loss, Tensor) else float(grad_loss)
        grad_pred = (2.0 * diff / size) * scale
        return Tensor(grad_pred, device="cpu")
