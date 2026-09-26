"""
Span Representation, Tabulators, Companions, and Conjoints for Lenses.
Based on Bryce Clarke (2022), Chapter 2, Section 2.4 and Chapter 3, Section 3.4.
"""

from typing import Any

from double_lenses.category.base import Category, Functor
from double_lenses.category.cofunctor import Cofunctor
from double_lenses.category.double_category import DoubleCell
from double_lenses.category.lens import DeltaLens


class Span:
    r"""
    A span of functors in Cat:
           X
         /   \
       ψ/     \g
       v       v
       A       B
    """

    def __init__(self, apex: Category, left_leg: Functor, right_leg: Functor):
        assert left_leg.source == apex, "Left leg source must be apex"
        assert right_leg.source == apex, "Right leg source must be apex"
        self.apex = apex
        self.left_leg = left_leg
        self.right_leg = right_leg
        self.source = left_leg.target
        self.target = right_leg.target

    def __repr__(self) -> str:
        return f"Span({self.source.name} <- {self.left_leg.name}- {self.apex.name} -{self.right_leg.name}-> {self.target.name})"


def tabulator_of_cofunctor(cof: Cofunctor) -> tuple[Category, Functor, Functor, DoubleCell]:
    """
    Proposition 3.5: The tabulator of a cofunctor (f, ϕ): A -> B is the category Λ(f, ϕ),
    equipped with projection functors:
      ψ: Λ(f, ϕ) -> A, ψ(a) = a, ψ(a, u) = ϕ(a, u)
      g: Λ(f, ϕ) -> B, g(a) = fa, g(a, u) = u
    and a cell in Cof:
          Λ(f, ϕ) ---- ψ ----> A
             |                  |
           1 |      θ_tab       | (f, ϕ)
             v                  v
          Λ(f, ϕ) ---- g ----> B
    """
    lambda_cat = cof.category_of_chosen_lifts()

    psi = Functor(
        name=f"ψ_{cof.name}",
        source=lambda_cat,
        target=cof.source,
        on_objects=lambda a: a,
        on_morphisms=lambda m: cof.lift(m[0], m[1]),
    )

    g = Functor(
        name=f"g_{cof.name}",
        source=lambda_cat,
        target=cof.target,
        on_objects=lambda a: cof.on_object(a),
        on_morphisms=lambda m: m[1],
    )

    return lambda_cat, psi, g


def span_representation_of_lens(lens: DeltaLens) -> Span:
    r"""
    Proposition 2.13: Every delta lens (f, ϕ): A -> B is represented by a commutative
    diagram of functors:
           Λ(f, ϕ)
          /       \
       ψ /         \ f ∘ ψ
        v           v
        A --------> B
              f
    where ψ is bijective-on-objects and f ∘ ψ is a discrete opfibration.
    """
    lambda_cat, psi, _ = tabulator_of_cofunctor(lens.cofunctor)
    right_leg = lens.forward.compose(psi)
    return Span(apex=lambda_cat, left_leg=psi, right_leg=right_leg)


def companion_of_functor(f: Functor) -> Cofunctor:
    """
    Proposition 3.3: A functor f: A -> B has a vertical companion in Cof
    if and only if f is a discrete opfibration.
    The companion cofunctor has object assignment f_0 and lifting γ(a, u)
    which sends (a, u: fa -> b) to the unique lift of u at a.
    """
    if not f.is_discrete_opfibration():
        raise ValueError(f"Functor {f.name} is not a discrete opfibration; cannot form companion.")

    def lift_fn(a: Any, u: Any) -> Any:
        # Find the unique lift w in hom(a, -) such that f(w) = u
        for w in f.source.morphisms:
            if f.source.dom(w) == a and f.on_morphism(w) == u:
                return w
        raise RuntimeError(f"Unique lift not found for u={u} at a={a}")

    return Cofunctor(
        name=f"({f.name})_*",
        source=f.source,
        target=f.target,
        on_objects=f.on_object,
        lift_fn=lift_fn,
    )


def conjoint_of_functor(f: Functor) -> Cofunctor:
    """
    Proposition 3.2: A functor f: A -> B has a vertical conjoint in Cof
    if and only if f is bijective-on-objects.
    The conjoint cofunctor f^*: B -> A has object assignment f_0^-1
    and lifting operation: (b, u: f^-1(b) -> a) ↦ f(u).
    """
    if not f.is_bijective_on_objects():
        raise ValueError(f"Functor {f.name} is not bijective-on-objects; cannot form conjoint.")

    inv_obj = {f.on_object(a): a for a in f.source.objects}

    return Cofunctor(
        name=f"({f.name})^*",
        source=f.target,
        target=f.source,
        on_objects=lambda b: inv_obj[b],
        lift_fn=lambda b, u: f.on_morphism(u),
    )
