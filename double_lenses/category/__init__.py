"""
Double Category of Lenses: Mathematical Core.
Based on Bryce Clarke (2022).
"""

from double_lenses.category.base import (
    Category,
    Functor,
    identity_functor,
)
from double_lenses.category.cofunctor import (
    Cofunctor,
    identity_cofunctor,
)
from double_lenses.category.double_category import (
    DoubleCategory,
    DoubleCategoryCof,
    DoubleCell,
)
from double_lenses.category.lens import (
    DeltaLens,
    identity_lens,
)
from double_lenses.category.monoidal import (
    product_category,
    swap_functor,
    tensor_cofunctor,
    tensor_double_cell,
    tensor_functor,
    tensor_lens,
)
from double_lenses.category.right_connected import (
    DoubleCategoryLens,
    RightConnectedCompletion,
)
from double_lenses.category.span_repr import (
    Span,
    companion_of_functor,
    conjoint_of_functor,
    span_representation_of_lens,
    tabulator_of_cofunctor,
)

__all__ = [
    "Category",
    "Functor",
    "identity_functor",
    "Cofunctor",
    "identity_cofunctor",
    "DeltaLens",
    "identity_lens",
    "DoubleCell",
    "DoubleCategory",
    "DoubleCategoryCof",
    "RightConnectedCompletion",
    "DoubleCategoryLens",
    "Span",
    "tabulator_of_cofunctor",
    "span_representation_of_lens",
    "companion_of_functor",
    "conjoint_of_functor",
    "product_category",
    "tensor_functor",
    "tensor_cofunctor",
    "tensor_lens",
    "tensor_double_cell",
    "swap_functor",
]
