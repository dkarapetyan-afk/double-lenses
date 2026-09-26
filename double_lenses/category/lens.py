"""
Delta Lenses and Vertical Lens Composition.
Based on Bryce Clarke (2022), Chapter 2, Section 2.1.
"""

from collections.abc import Callable
from typing import Any

from double_lenses.category.base import Category, Functor, identity_functor
from double_lenses.category.cofunctor import Cofunctor


class DeltaLens:
    """
    Definition 2.1: A delta lens (f, ϕ): A -> B consists of:
      - A functor f: A -> B
      - A lifting operation ϕ: (a ∈ A_0, u: fa -> b ∈ B_1) ↦ ϕ(a, u): a -> p(a, u)
    satisfying axioms:
      (L1) f(ϕ(a, u)) = u
      (L2) ϕ(a, 1_{fa}) = 1_a
      (L3) ϕ(a, v ∘ u) = ϕ(p(a, u), v) ∘ ϕ(a, u)
    """

    def __init__(
        self,
        name: str,
        forward_functor: Functor,
        lift_fn: Callable[[Any, Any], Any],
    ):
        self.name = name
        self.forward = forward_functor
        self.source: Category = forward_functor.source
        self.target: Category = forward_functor.target
        self._lift_fn = lift_fn
        # Lemma 2.3: Canonical underlying cofunctor
        self.cofunctor = Cofunctor(
            name=f"Cof({name})",
            source=self.source,
            target=self.target,
            on_objects=forward_functor.on_object,
            lift_fn=lift_fn,
        )

    def get(self, a: Any) -> Any:
        """Functorial forward evaluation / get."""
        return self.forward.on_object(a)

    def lift(self, a: Any, u: Any) -> Any:
        """Computes ϕ(a, u): a -> p(a, u)."""
        fa = self.get(a)
        if self.target.dom(u) != fa:
            raise ValueError(f"Morphism {u} domain {self.target.dom(u)} does not match f(a)={fa}")
        return self._lift_fn(a, u)

    def put(self, a: Any, u: Any) -> Any:
        """p(a, u) := cod(ϕ(a, u))."""
        phi = self.lift(a, u)
        return self.source.cod(phi)

    def compose(self, other: "DeltaLens") -> "DeltaLens":
        """
        Vertical composition of lenses: self ∘ other: A -> C (Clarke Eq. 2.1)
        where other: A -> B and self: B -> C.
        Functor is self.forward ∘ other.forward.
        Lifting operation is:
        (a, u: gfa -> c) ↦ other.lift(a, self.lift(other.get(a), u))
        """
        if self.source != other.target:
            raise ValueError(f"Cannot compose lenses {self.name} and {other.name}: target mismatch")

        comp_forward = self.forward.compose(other.forward)

        def composed_lift(a: Any, u: Any) -> Any:
            fa = other.get(a)
            gamma_fa_u = self.lift(fa, u)
            return other.lift(a, gamma_fa_u)

        return DeltaLens(
            name=f"{self.name} ∘ {other.name}",
            forward_functor=comp_forward,
            lift_fn=composed_lift,
        )

    def validate_axioms(self) -> tuple[bool, str | None]:
        """
        Validates axioms L1, L2, L3:
          (L1) f(ϕ(a, u)) = u
          (L2) ϕ(a, 1_{fa}) = 1_a
          (L3) ϕ(a, v ∘ u) = ϕ(p(a, u), v) ∘ ϕ(a, u)
        """
        # Validate forward functor
        if not self.forward.validate():
            return False, "Underlying forward functor fails functor axioms"

        # L2: Identity
        for a in self.source.objects:
            fa = self.get(a)
            id_fa = self.target.identity(fa)
            phi_id = self.lift(a, id_fa)
            id_a = self.source.identity(a)
            if phi_id != id_a:
                return False, f"Axiom L2 failed at a={a}: ϕ(a, 1_fa)={phi_id} != 1_a={id_a}"

        # L1 & L3: Lifts over u and composition
        for a in self.source.objects:
            fa = self.get(a)
            for u in self.target.morphisms:
                if self.target.dom(u) == fa:
                    phi_u = self.lift(a, u)
                    # L1 check
                    if self.forward.on_morphism(phi_u) != u:
                        return False, f"Axiom L1 failed: f(ϕ(a, u))={self.forward.on_morphism(phi_u)} != u={u}"

                    pa_u = self.source.cod(phi_u)
                    for v in self.target.morphisms:
                        if self.target.dom(v) == self.target.cod(u):
                            vu = self.target.compose(v, u)
                            phi_vu = self.lift(a, vu)
                            phi_v = self.lift(pa_u, v)
                            comp_lifts = self.source.compose(phi_v, phi_u)
                            if phi_vu != comp_lifts:
                                return (
                                    False,
                                    f"Axiom L3 failed: ϕ(a, v∘u)={phi_vu} != ϕ(p(a,u), v) ∘ ϕ(a,u)={comp_lifts}",
                                )
        return True, None

    def category_of_chosen_lifts(self) -> Category:
        """Category of chosen lifts Λ(f, ϕ)."""
        return self.cofunctor.category_of_chosen_lifts()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DeltaLens):
            return False
        return self.name == other.name and self.source == other.source and self.target == other.target

    def __repr__(self) -> str:
        return f"DeltaLens({self.name}: {self.source.name} ⇄ {self.target.name})"


def identity_lens(cat: Category) -> DeltaLens:
    """Identity lens on category A."""
    return DeltaLens(
        name=f"1^L_{cat.name}",
        forward_functor=identity_functor(cat),
        lift_fn=lambda a, u: u,
    )
