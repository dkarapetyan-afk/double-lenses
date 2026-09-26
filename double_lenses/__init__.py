"""
Double Lenses: A mathematical foundation based on Bryce Clarke's PhD thesis
'The double category of lenses' (2022) and distributed neural network runtime
for forward and adjoint computations in large models (Mixtral, DeepSeek).
"""

__version__ = "0.1.0"

from double_lenses.category import (
    Category,
    Functor,
    Cofunctor,
    DeltaLens,
    DoubleCategory,
    DoubleCell,
    RightConnectedCompletion,
)

__all__ = [
    "Category",
    "Functor",
    "Cofunctor",
    "DeltaLens",
    "DoubleCategory",
    "DoubleCell",
    "RightConnectedCompletion",
]
