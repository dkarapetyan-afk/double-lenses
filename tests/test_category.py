"""
Unit tests for Category Theory Core:
  - Categories, Functors, Discrete Opfibrations
  - Cofunctors & Axioms (C1, C2, C3)
  - Category of Chosen Lifts Λ(f, ϕ)
  - Delta Lenses & Axioms (L1, L2, L3)
  - Double Category Interchange Law
Based on Bryce Clarke (2022).
"""

from double_lenses.category.base import Category, Functor, identity_functor
from double_lenses.category.cofunctor import Cofunctor, identity_cofunctor
from double_lenses.category.double_category import (
    DoubleCategory,
    DoubleCell,
)
from double_lenses.category.lens import DeltaLens, identity_lens


def create_sample_categories():
    """
    Creates finite test categories A and B:
      A: 3 objects (a0, a1, a2), morphisms: id_0, id_1, id_2, f01: a0->a1, f12: a1->a2, f02: a0->a2 (f12 ∘ f01 = f02)
      B: 2 objects (b0, b1), morphisms: id_b0, id_b1, u: b0->b1
    """
    cat_a = Category(
        name="A",
        objects=["a0", "a1", "a2"],
        morphisms=["id_a0", "id_a1", "id_a2", "f01", "f12", "f02"],
        dom_fn=lambda m: {"id_a0": "a0", "id_a1": "a1", "id_a2": "a2", "f01": "a0", "f12": "a1", "f02": "a0"}[m],
        cod_fn=lambda m: {"id_a0": "a0", "id_a1": "a1", "id_a2": "a2", "f01": "a1", "f12": "a2", "f02": "a2"}[m],
        id_fn=lambda o: f"id_{o}",
        compose_fn=lambda g, f: "f02" if (g, f) == ("f12", "f01") else (f if g.startswith("id") else g),
    )

    cat_b = Category(
        name="B",
        objects=["b0", "b1"],
        morphisms=["id_b0", "id_b1", "u"],
        dom_fn=lambda m: {"id_b0": "b0", "id_b1": "b1", "u": "b0"}[m],
        cod_fn=lambda m: {"id_b0": "b0", "id_b1": "b1", "u": "b1"}[m],
        id_fn=lambda o: f"id_{o}",
        compose_fn=lambda g, f: f if g.startswith("id") else g,
    )
    return cat_a, cat_b


def test_category_axioms():
    cat_a, cat_b = create_sample_categories()
    assert cat_a.validate()
    assert cat_b.validate()


def test_functor_axioms():
    cat_a, cat_b = create_sample_categories()
    f = Functor(
        name="F",
        source=cat_a,
        target=cat_b,
        on_objects=lambda a: "b0" if a in ("a0", "a1") else "b1",
        on_morphisms=lambda m: "id_b0" if m in ("id_a0", "id_a1", "f01") else ("u" if m in ("f12", "f02") else "id_b1"),
    )
    assert f.validate()


def test_cofunctor_axioms_and_lifts():
    cat_a, cat_b = create_sample_categories()

    # Define a valid cofunctor (f, ϕ): A -> B
    # f maps a0 -> b0, a1 -> b0, a2 -> b1
    # u: b0 -> b1 has lift at a0 as f02, and at a1 as f12
    def lift_fn(a, u):
        if u == "id_b0":
            return cat_a.identity(a)
        if u == "id_b1":
            return cat_a.identity(a)
        if u == "u":
            if a == "a0":
                return "f02"
            elif a == "a1":
                return "f12"
        raise ValueError(f"No lift for ({a}, {u})")

    cof = Cofunctor(
        name="Cof_AB",
        source=cat_a,
        target=cat_b,
        on_objects=lambda a: "b0" if a in ("a0", "a1") else "b1",
        lift_fn=lift_fn,
    )

    valid, err = cof.validate_axioms()
    assert valid, f"Cofunctor validation failed: {err}"

    # Verify category of chosen lifts Λ(f, ϕ)
    lambda_cat = cof.category_of_chosen_lifts()
    assert lambda_cat.validate()
    assert len(lambda_cat.objects) == len(cat_a.objects)


def test_delta_lens_axioms():
    cat_a, cat_b = create_sample_categories()

    fwd = Functor(
        name="F_lens",
        source=cat_a,
        target=cat_b,
        on_objects=lambda a: "b0" if a in ("a0", "a1") else "b1",
        on_morphisms=lambda m: "id_b0" if m in ("id_a0", "id_a1", "f01") else ("u" if m in ("f12", "f02") else "id_b1"),
    )

    def lift_fn(a, u):
        if u.startswith("id_"):
            return cat_a.identity(a)
        if u == "u":
            return "f02" if a == "a0" else "f12"
        raise ValueError(f"No lift for ({a}, {u})")

    lens = DeltaLens(name="Lens_AB", forward_functor=fwd, lift_fn=lift_fn)
    valid, err = lens.validate_axioms()
    assert valid, f"Lens validation failed: {err}"


def test_lens_vertical_composition():
    cat_a, cat_b = create_sample_categories()

    # Identity lenses
    id_a = identity_lens(cat_a)
    _id_b = identity_lens(cat_b)

    fwd = Functor(
        name="F",
        source=cat_a,
        target=cat_b,
        on_objects=lambda a: "b0" if a in ("a0", "a1") else "b1",
        on_morphisms=lambda m: "id_b0" if m in ("id_a0", "id_a1", "f01") else ("u" if m in ("f12", "f02") else "id_b1"),
    )
    lens = DeltaLens(
        name="Lens_AB",
        forward_functor=fwd,
        lift_fn=lambda a, u: "f02" if (a, u) == ("a0", "u") else ("f12" if (a, u) == ("a1", "u") else f"id_{a}"),
    )

    # Compose with identity: lens ∘ id_a == lens
    comp = lens.compose(id_a)
    assert comp.get("a0") == lens.get("a0")
    assert comp.lift("a0", "u") == lens.lift("a0", "u")


def test_double_category_interchange_law():
    cat_a, _ = create_sample_categories()
    id_f = identity_functor(cat_a)
    id_c = identity_cofunctor(cat_a)

    # 4 cells in 2x2 grid
    c11 = DoubleCell("c11", id_f, id_f, id_c, id_c)
    c12 = DoubleCell("c12", id_f, id_f, id_c, id_c)
    c21 = DoubleCell("c21", id_f, id_f, id_c, id_c)
    c22 = DoubleCell("c22", id_f, id_f, id_c, id_c)

    interchange_holds = DoubleCategory.verify_interchange(c11, c12, c21, c22)
    assert interchange_holds
