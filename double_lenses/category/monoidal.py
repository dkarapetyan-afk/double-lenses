"""
Symmetric Monoidal Structure on Categories, Cofunctors, and Lenses.
Models parallel composition (tensor product ⊗) for parallel layers,
multi-head attention branches, and multi-GPU tensor sharding.
"""

from typing import Any

from double_lenses.category.base import Category, Functor
from double_lenses.category.cofunctor import Cofunctor
from double_lenses.category.double_category import DoubleCell
from double_lenses.category.lens import DeltaLens


def product_category(c1: Category, c2: Category) -> Category:
    """Computes the Cartesian product / tensor product category C1 ⊗ C2."""
    objects = [(o1, o2) for o1 in c1.objects for o2 in c2.objects]
    morphisms = [(m1, m2) for m1 in c1.morphisms for m2 in c2.morphisms]

    def dom_fn(m: tuple[Any, Any]) -> tuple[Any, Any]:
        return (c1.dom(m[0]), c2.dom(m[1]))

    def cod_fn(m: tuple[Any, Any]) -> tuple[Any, Any]:
        return (c1.cod(m[0]), c2.cod(m[1]))

    def id_fn(o: tuple[Any, Any]) -> tuple[Any, Any]:
        return (c1.identity(o[0]), c2.identity(o[1]))

    def compose_fn(g: tuple[Any, Any], f: tuple[Any, Any]) -> tuple[Any, Any]:
        return (c1.compose(g[0], f[0]), c2.compose(g[1], f[1]))

    return Category(
        name=f"{c1.name} ⊗ {c2.name}",
        objects=objects,
        morphisms=morphisms,
        dom_fn=dom_fn,
        cod_fn=cod_fn,
        id_fn=id_fn,
        compose_fn=compose_fn,
    )


def tensor_functor(f1: Functor, f2: Functor) -> Functor:
    """Parallel tensor product of functors: f1 ⊗ f2: A1 ⊗ A2 -> B1 ⊗ B2."""
    source = product_category(f1.source, f2.source)
    target = product_category(f1.target, f2.target)

    return Functor(
        name=f"{f1.name} ⊗ {f2.name}",
        source=source,
        target=target,
        on_objects=lambda pair: (f1.on_object(pair[0]), f2.on_object(pair[1])),
        on_morphisms=lambda pair: (f1.on_morphism(pair[0]), f2.on_morphism(pair[1])),
    )


def tensor_cofunctor(c1: Cofunctor, c2: Cofunctor) -> Cofunctor:
    """Parallel tensor product of cofunctors: c1 ⊗ c2: A1 ⊗ A2 -> B1 ⊗ B2."""
    source = product_category(c1.source, c2.source)
    target = product_category(c1.target, c2.target)

    def lift_fn(a_pair: tuple[Any, Any], u_pair: tuple[Any, Any]) -> tuple[Any, Any]:
        a1, a2 = a_pair
        u1, u2 = u_pair
        return (c1.lift(a1, u1), c2.lift(a2, u2))

    return Cofunctor(
        name=f"{c1.name} ⊗ {c2.name}",
        source=source,
        target=target,
        on_objects=lambda pair: (c1.on_object(pair[0]), c2.on_object(pair[1])),
        lift_fn=lift_fn,
    )


def tensor_lens(l1: DeltaLens, l2: DeltaLens) -> DeltaLens:
    """Parallel tensor product of delta lenses: l1 ⊗ l2: A1 ⊗ A2 -> B1 ⊗ B2."""
    fwd = tensor_functor(l1.forward, l2.forward)

    def lift_fn(a_pair: tuple[Any, Any], u_pair: tuple[Any, Any]) -> tuple[Any, Any]:
        a1, a2 = a_pair
        u1, u2 = u_pair
        return (l1.lift(a1, u1), l2.lift(a2, u2))

    return DeltaLens(
        name=f"{l1.name} ⊗ {l2.name}",
        forward_functor=fwd,
        lift_fn=lift_fn,
    )


def tensor_double_cell(cell1: DoubleCell, cell2: DoubleCell) -> DoubleCell:
    """Parallel tensor product of double cells (2-cells)."""
    top = tensor_functor(cell1.top, cell2.top)
    bottom = tensor_functor(cell1.bottom, cell2.bottom)

    # Check whether left/right are lenses or cofunctors
    if isinstance(cell1.left, DeltaLens) and isinstance(cell2.left, DeltaLens):
        left = tensor_lens(cell1.left, cell2.left)
        right = tensor_lens(cell1.right, cell2.right)
    else:
        left = tensor_cofunctor(cell1.left, cell2.left)
        right = tensor_cofunctor(cell1.right, cell2.right)

    return DoubleCell(
        name=f"{cell1.name} ⊗ {cell2.name}",
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        payload=(cell1.payload, cell2.payload),
    )


def swap_functor(c1: Category, c2: Category) -> Functor:
    """Symmetry braiding isomorphism: σ: C1 ⊗ C2 -> C2 ⊗ C1."""
    source = product_category(c1, c2)
    target = product_category(c2, c1)
    return Functor(
        name=f"σ_{{{c1.name}, {c2.name}}}",
        source=source,
        target=target,
        on_objects=lambda pair: (pair[1], pair[0]),
        on_morphisms=lambda pair: (pair[1], pair[0]),
    )
