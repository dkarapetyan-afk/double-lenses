"""
Unit tests for Span Representation, Tabulators, Conjoints and Companions.
Based on Bryce Clarke (2022), Chapter 2, Section 2.4 and Chapter 3, Section 3.4.
"""

from double_lenses.category.base import Category, Functor, identity_functor
from double_lenses.category.cofunctor import Cofunctor
from double_lenses.category.lens import DeltaLens
from double_lenses.category.span_repr import (
    tabulator_of_cofunctor,
    span_representation_of_lens,
    companion_of_functor,
    conjoint_of_functor,
)


def test_span_representation_of_lens():
    cat = Category(
        name="C",
        objects=["c0", "c1"],
        morphisms=["id_c0", "id_c1", "e"],
        dom_fn=lambda m: "c0" if m in ("id_c0", "e") else "c1",
        cod_fn=lambda m: "c0" if m == "id_c0" else "c1",
        id_fn=lambda o: f"id_{o}",
        compose_fn=lambda g, f: f if g.startswith("id") else g,
    )

    fwd = identity_functor(cat)
    lens = DeltaLens("IdLens", fwd, lambda a, u: u)

    span = span_representation_of_lens(lens)
    assert span.source == cat
    assert span.target == cat
    # ψ is bijective-on-objects
    assert span.left_leg.is_bijective_on_objects()
    # right_leg is a discrete opfibration
    assert span.right_leg.is_discrete_opfibration()


def test_companions_and_conjoints():
    cat = Category(
        name="C",
        objects=["c0"],
        morphisms=["id_c0"],
        dom_fn=lambda m: "c0",
        cod_fn=lambda m: "c0",
        id_fn=lambda o: "id_c0",
        compose_fn=lambda g, f: "id_c0",
    )
    fwd = identity_functor(cat)

    # Identity functor is both bijective-on-objects and discrete opfibration
    companion = companion_of_functor(fwd)
    conjoint = conjoint_of_functor(fwd)

    assert companion.on_object("c0") == "c0"
    assert conjoint.on_object("c0") == "c0"
