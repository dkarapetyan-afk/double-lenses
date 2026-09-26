"""
Categorical Reverse Derivative & Parameterized Lens Autodiff Engine.
"""

from double_lenses.autodiff.layers import (
    EmbeddingLens,
    LinearLens,
    RMSNormLens,
    RoPELens,
    SiLULens,
    SoftmaxLens,
    SwiGLULens,
)
from double_lenses.autodiff.loss import (
    CrossEntropyLossLens,
    MSELossLens,
)
from double_lenses.autodiff.param_lens import (
    ComposedLens,
    LensContext,
    ParameterizedLens,
    SequentialLens,
)
from double_lenses.autodiff.tensor import (
    Tensor,
    ones,
    randn,
    zeros,
)

__all__ = [
    "Tensor",
    "zeros",
    "ones",
    "randn",
    "LensContext",
    "ParameterizedLens",
    "ComposedLens",
    "SequentialLens",
    "LinearLens",
    "RMSNormLens",
    "SiLULens",
    "SwiGLULens",
    "SoftmaxLens",
    "RoPELens",
    "EmbeddingLens",
    "CrossEntropyLossLens",
    "MSELossLens",
]
