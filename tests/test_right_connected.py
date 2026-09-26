"""
Unit tests for Right-Connected Completion and Theorem 3.21: Lens ≅ Γ(Cof).
Based on Bryce Clarke (2022), Chapter 3, Section 3.2 and 3.3.
"""

from double_lenses.category.base import Category, Functor, identity_functor
from double_lenses.category.cofunctor import Cofunctor, identity_cofunctor
from double_lenses.category.lens import DeltaLens
from double_lenses.category.right_connected import (
    RightConnectedCompletion,
    DoubleCategoryLens,
)


def test_theorem_3_21_lens_isomorphism():
    cat_a = Category(
        name="A",
        objects=["x", "y"],
        morphisms=["id_x", "id_y", "m"],
        dom_fn=lambda m: "x" if m in ("id_x", "m") else "y",
        cod_fn=lambda m: "x" if m == "id_x" else "y",
        id_fn=lambda o: f"id_{o}",
        compose_fn=lambda g, f: f if g.startswith("id") else g,
    )

    fwd = identity_functor(cat_a)
    lens = DeltaLens(name="IdLens", forward_functor=fwd, lift_fn=lambda a, u: u)

    rc = RightConnectedCompletion()

    # Convert lens to vertical arrow in Γ(Cof)
    left_cof, cell, top_fun = rc.lens_to_vertical_arrow(lens)

    # In Γ(Cof), the cell condition is f(ϕ(a, u)) = u (Lens Axiom L1)
    assert top_fun == fwd
    assert left_cof.name.startswith("Cof")
    assert cell.bottom.name == f"1_{cat_a.name}"

    # Convert back to lens
    reconstructed_lens = rc.vertical_arrow_to_lens(left_cof, cell, top_fun)
    assert reconstructed_lens.get("x") == "x"
    assert reconstructed_lens.lift("x", "m") == "m"


def test_lens_double_category_cells():
    cat_a = Category(
        name="A",
        objects=["x"],
        morphisms=["id_x"],
        dom_fn=lambda m: "x",
        cod_fn=lambda m: "x",
        id_fn=lambda o: "id_x",
        compose_fn=lambda g, f: "id_x",
    )

    dbl_lens = DoubleCategoryLens()
    id_f = identity_functor(cat_a)
    id_l = DeltaLens("IdLens", id_f, lambda a, u: u)

    cell = dbl_lens.make_cell(
        name="id_cell",
        top=id_f,
        bottom=id_f,
        left=id_l,
        right=id_l,
    )
    assert cell.top == id_f
    assert cell.bottom == id_f
