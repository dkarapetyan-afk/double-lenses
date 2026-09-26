"""
Categorical Reverse Derivative & Parameterized Lens Autodiff Engine.
"""

from double_lenses.autodiff.tensor import (
    Tensor,
    zeros,
    ones,
    randn,
)
from double_lenses.autodiff.param_lens import (
    LensContext,
    ParameterizedLens,
    ComposedLens,
    SequentialLens,
)
from double_lenses.autodiff.layers import (
    LinearLens,
    RMSNormLens,
    SiLULens,
    SwiGLULens,
    SoftmaxLens,
    RoPELens,
    EmbeddingLens,
)
from double_lenses.autodiff.loss import (
    CrossEntropyLossLens,
    MSELossLens,
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
