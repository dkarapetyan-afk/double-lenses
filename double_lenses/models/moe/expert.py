"""
Expert Lens implementation wrapping SwiGLU Feed-Forward Networks.
Supports placement on specific cluster hardware (GPU/CPU) and staging.
"""

from typing import Optional
from double_lenses.autodiff.tensor import Tensor
from double_lenses.autodiff.param_lens import LensContext, ParameterizedLens
from double_lenses.autodiff.layers import SwiGLULens


class ExpertLens(ParameterizedLens):
    """
    A single SwiGLU expert lens:
      SwiGLU(x) = (silu(x W_gate^T) * (x W_up^T)) W_down^T
    """
    def __init__(self, name: str, in_dim: int, hidden_dim: int, out_dim: Optional[int] = None, expert_id: int = 0):
        super().__init__(name=name)
        self.expert_id = expert_id
        self.swiglu = SwiGLULens(name=f"{name}.swiglu", in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)

        for k, v in self.swiglu.parameters.items():
            self.parameters[f"swiglu.{k}"] = v

    def forward(self, ctx: LensContext, x: Tensor) -> Tensor:
        return self.swiglu.forward(ctx, x)

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        return self.swiglu.adjoint(ctx, grad_y)
