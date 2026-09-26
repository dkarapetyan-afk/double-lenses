"""
Base categorical definitions: Categories, Functors, Natural Transformations,
Discrete Opfibrations, and Bijective-on-Objects Functors.
Based on Bryce Clarke (2022), Chapter 2 and Appendix A.
"""

from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Set, Tuple, TypeVar

O = TypeVar('O')  # Object type
M = TypeVar('M')  # Morphism type


class Category(Generic[O, M]):
    """
    A small category consisting of objects and morphisms with identities and composition.
    """
    def __init__(
        self,
        name: str,
        objects: Iterable[O],
        morphisms: Iterable[M],
        dom_fn: Callable[[M], O],
        cod_fn: Callable[[M], O],
        id_fn: Callable[[O], M],
        compose_fn: Callable[[M, M], M],  # compose(g, f) = g ∘ f
    ):
        self.name = name
        self.objects: Set[O] = set(objects)
        self.morphisms: Set[M] = set(morphisms)
        self._dom_fn = dom_fn
        self._cod_fn = cod_fn
        self._id_fn = id_fn
        self._compose_fn = compose_fn

    def dom(self, m: M) -> O:
        return self._dom_fn(m)

    def cod(self, m: M) -> O:
        return self._cod_fn(m)

    def identity(self, o: O) -> M:
        if o not in self.objects:
            raise ValueError(f"Object {o} is not in category {self.name}")
        return self._id_fn(o)

    def compose(self, g: M, f: M) -> M:
        """Computes g ∘ f (first f, then g). Requires dom(g) == cod(f)."""
        if self.dom(g) != self.cod(f):
            raise ValueError(
                f"Cannot compose {g} (dom={self.dom(g)}) and {f} (cod={self.cod(f)}) in {self.name}"
            )
        return self._compose_fn(g, f)

    def hom(self, a: O, b: O) -> List[M]:
        """Returns all morphisms from a to b."""
        return [m for m in self.morphisms if self.dom(m) == a and self.cod(m) == b]

    def validate(self) -> bool:
        """Validates category axioms: unitality and associativity on finite elements."""
        for o in self.objects:
            ida = self.identity(o)
            if self.dom(ida) != o or self.cod(ida) != o:
                return False

        for f in self.morphisms:
            ida = self.identity(self.dom(f))
            idb = self.identity(self.cod(f))
            if self.compose(f, ida) != f or self.compose(idb, f) != f:
                return False

        # Associativity check for composable triples
        for f in self.morphisms:
            for g in self.hom(self.cod(f), None):  # all g with dom(g) == cod(f)
                gf = self.compose(g, f)
                for h in self.hom(self.cod(g), None):
                    h_gf = self.compose(h, gf)
                    hg_f = self.compose(self.compose(h, g), f)
                    if h_gf != hg_f:
                        return False
        return True

    def __repr__(self) -> str:
        return f"Category({self.name}, |Ob|={len(self.objects)}, |Mor|={len(self.morphisms)})"


class Functor:
    """
    A functor F: A -> B between categories A and B.
    """
    def __init__(
        self,
        name: str,
        source: Category,
        target: Category,
        on_objects: Callable[[Any], Any],
        on_morphisms: Callable[[Any], Any],
    ):
        self.name = name
        self.source = source
        self.target = target
        self._on_objects = on_objects
        self._on_morphisms = on_morphisms

    def on_object(self, a: Any) -> Any:
        return self._on_objects(a)

    def on_morphism(self, u: Any) -> Any:
        return self._on_morphisms(u)

    def __call__(self, x: Any) -> Any:
        if x in self.source.objects:
            return self.on_object(x)
        return self.on_morphism(x)

    def compose(self, other: 'Functor') -> 'Functor':
        """Returns self ∘ other (first other, then self)."""
        if self.source != other.target:
            raise ValueError(f"Cannot compose functor {self.name} with {other.name}: target mismatch")
        return Functor(
            name=f"{self.name} ∘ {other.name}",
            source=other.source,
            target=self.target,
            on_objects=lambda a: self.on_object(other.on_object(a)),
            on_morphisms=lambda u: self.on_morphism(other.on_morphism(u)),
        )

    def is_bijective_on_objects(self) -> bool:
        """Definition 2.8: Checks if F_0: A_0 -> B_0 is bijective."""
        target_objs = set(self.target.objects)
        mapped = [self.on_object(a) for a in self.source.objects]
        return len(mapped) == len(set(mapped)) and set(mapped) == target_objs

    def is_discrete_opfibration(self) -> bool:
        """
        Definition 2.9: Checks if for all a in A and u: F(a) -> b in B,
        there is a UNIQUE morphism w: a -> a' in A such that F(w) = u.
        """
        for a in self.source.objects:
            fa = self.on_object(a)
            for u in self.target.morphisms:
                if self.target.dom(u) == fa:
                    # look for lifts in hom(a, -)
                    lifts = []
                    for w in self.source.morphisms:
                        if self.source.dom(w) == a and self.on_morphism(w) == u:
                            lifts.append(w)
                    if len(lifts) != 1:
                        return False
        return True

    def validate(self) -> bool:
        """Verifies identity and composition preservation."""
        for a in self.source.objects:
            ida = self.source.identity(a)
            if self.on_morphism(ida) != self.target.identity(self.on_object(a)):
                return False
        for f in self.source.morphisms:
            for g in self.source.hom(self.source.cod(f), None):
                gf = self.source.compose(g, f)
                if self.on_morphism(gf) != self.target.compose(self.on_morphism(g), self.on_morphism(f)):
                    return False
        return True

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Functor):
            return False
        return self.name == other.name and self.source == other.source and self.target == other.target

    def __repr__(self) -> str:
        return f"Functor({self.name}: {self.source.name} -> {self.target.name})"


def identity_functor(cat: Category) -> Functor:
    """Returns the identity functor 1_A on category A."""
    return Functor(
        name=f"1_{cat.name}",
        source=cat,
        target=cat,
        on_objects=lambda a: a,
        on_morphisms=lambda u: u,
    )
