"""
Cofunctors and the Category of Chosen Lifts Λ(f, ϕ).
Based on Bryce Clarke (2022), Chapter 2, Section 2.2 and 2.3.
"""

from collections.abc import Callable
from typing import Any

from double_lenses.category.base import Category


class Cofunctor:
    """
    Definition 2.2: A cofunctor (f, ϕ): A -> B consists of:
      - An object assignment f: A_0 -> B_0
      - A lifting operation ϕ: (a ∈ A_0, u: fa -> b ∈ B_1) ↦ ϕ(a, u): a -> p(a, u)
    satisfying axioms (C1), (C2), and (C3).
    """

    def __init__(
        self,
        name: str,
        source: Category,
        target: Category,
        on_objects: Callable[[Any], Any],
        lift_fn: Callable[[Any, Any], Any],  # lift_fn(a, u) -> morphism in source
    ):
        self.name = name
        self.source = source
        self.target = target
        self._on_objects = on_objects
        self._lift_fn = lift_fn

    def on_object(self, a: Any) -> Any:
        return self._on_objects(a)

    def lift(self, a: Any, u: Any) -> Any:
        """Computes ϕ(a, u): a -> p(a, u)."""
        fa = self.on_object(a)
        if self.target.dom(u) != fa:
            raise ValueError(f"Morphism {u} domain {self.target.dom(u)} does not match f(a)={fa}")
        return self._lift_fn(a, u)

    def put(self, a: Any, u: Any) -> Any:
        """p(a, u) := cod(ϕ(a, u)). In lens literature this is 'put'."""
        phi = self.lift(a, u)
        return self.source.cod(phi)

    def compose(self, other: "Cofunctor") -> "Cofunctor":
        """
        Vertical composition of cofunctors: self ∘ other: A -> C
        where other: A -> B and self: B -> C.
        (g, γ) ∘ (f, ϕ) has object map g(f(a)) and lift:
        (a, u: gfa -> c) ↦ ϕ(a, γ(fa, u))
        """
        if self.source != other.target:
            raise ValueError(f"Cannot compose cofunctors {self.name} and {other.name}: target mismatch")

        def composed_lift(a: Any, u: Any) -> Any:
            fa = other.on_object(a)
            gamma = self.lift(fa, u)
            return other.lift(a, gamma)

        return Cofunctor(
            name=f"{self.name} ∘ {other.name}",
            source=other.source,
            target=self.target,
            on_objects=lambda a: self.on_object(other.on_object(a)),
            lift_fn=composed_lift,
        )

    def validate_axioms(self) -> tuple[bool, str | None]:
        """
        Validates axioms C1, C2, C3:
          (C1) f(p(a, u)) = cod(u)
          (C2) ϕ(a, 1_{fa}) = 1_a
          (C3) ϕ(a, v ∘ u) = ϕ(p(a, u), v) ∘ ϕ(a, u)
        """
        # C2: Identity preservation
        for a in self.source.objects:
            fa = self.on_object(a)
            id_fa = self.target.identity(fa)
            phi_id = self.lift(a, id_fa)
            id_a = self.source.identity(a)
            if phi_id != id_a:
                return False, f"Axiom C2 failed at a={a}: ϕ(a, 1_{{fa}})={phi_id} != 1_a={id_a}"

        # C1 & C3: Composition and codomain preservation
        for a in self.source.objects:
            fa = self.on_object(a)
            for u in self.target.morphisms:
                if self.target.dom(u) == fa:
                    phi_u = self.lift(a, u)
                    pa_u = self.source.cod(phi_u)
                    # C1 check
                    if self.on_object(pa_u) != self.target.cod(u):
                        return (
                            False,
                            f"Axiom C1 failed: f(p(a, u))={self.on_object(pa_u)} != cod(u)={self.target.cod(u)}",
                        )

                    for v in self.target.morphisms:
                        if self.target.dom(v) == self.target.cod(u):
                            vu = self.target.compose(v, u)
                            phi_vu = self.lift(a, vu)
                            phi_v = self.lift(pa_u, v)
                            comp_lifts = self.source.compose(phi_v, phi_u)
                            if phi_vu != comp_lifts:
                                return (
                                    False,
                                    f"Axiom C3 failed: ϕ(a, v∘u)={phi_vu} != ϕ(p(a,u), v) ∘ ϕ(a,u)={comp_lifts}",
                                )
        return True, None

    def category_of_chosen_lifts(self) -> "Category":
        """
        Proposition 2.6: Given a cofunctor (f, ϕ): A -> B, constructs Λ(f, ϕ).
        Objects are same as A.
        Morphisms are pairs (a, u: fa -> b).
        dom(a, u) = a, cod(a, u) = p(a, u).
        compose((p(a, u), v), (a, u)) = (a, v ∘ u).
        """
        objects = list(self.source.objects)
        morphisms = []
        for a in self.source.objects:
            fa = self.on_object(a)
            for u in self.target.morphisms:
                if self.target.dom(u) == fa:
                    morphisms.append((a, u))

        def dom_fn(m: tuple[Any, Any]) -> Any:
            return m[0]

        def cod_fn(m: tuple[Any, Any]) -> Any:
            a, u = m
            return self.put(a, u)

        def id_fn(a: Any) -> tuple[Any, Any]:
            return (a, self.target.identity(self.on_object(a)))

        def compose_fn(g: tuple[Any, Any], f: tuple[Any, Any]) -> tuple[Any, Any]:
            # g = (p(a, u), v), f = (a, u)
            a, u = f
            pa_u, v = g
            return (a, self.target.compose(v, u))

        return Category(
            name=f"Λ({self.name})",
            objects=objects,
            morphisms=morphisms,
            dom_fn=dom_fn,
            cod_fn=cod_fn,
            id_fn=id_fn,
            compose_fn=compose_fn,
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Cofunctor):
            return False
        return self.name == other.name and self.source == other.source and self.target == other.target

    def __repr__(self) -> str:
        return f"Cofunctor({self.name}: {self.source.name} ⇸ {self.target.name})"


def identity_cofunctor(cat: Category) -> Cofunctor:
    """Identity cofunctor on category A: (1_A0, π) where π(a, u: a -> a') = u."""
    return Cofunctor(
        name=f"1^c_{cat.name}",
        source=cat,
        target=cat,
        on_objects=lambda a: a,
        lift_fn=lambda a, u: u,
    )
