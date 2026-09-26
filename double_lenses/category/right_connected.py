"""
Right-Connected Double Categories and the Right-Connected Completion Γ(D).
Formalizes Theorem 3.21: Lens ≅ Γ(Cof).
Based on Bryce Clarke (2022), Chapter 3, Section 3.2 and 3.3.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from double_lenses.category.base import Category, Functor, identity_functor
from double_lenses.category.cofunctor import Cofunctor, identity_cofunctor
from double_lenses.category.lens import DeltaLens
from double_lenses.category.double_category import DoubleCell, DoubleCategory, DoubleCategoryCof


class RightConnectedCompletion(DoubleCategory):
    """
    Definition 3.12: The right-connected completion of a double category D, denoted Γ(D).
      - Objects: Objects of D (categories)
      - Horizontal arrows: Horizontal arrows of D (functors)
      - Vertical arrows: Cells in D of the form:
              f'
          A -----> B
          |        |
        f |   α    | 1_B
          v        v
          B -----> B
             1_B
        represented as triples (f, α, f').
      - Cells: Cells θ in D satisfying the right-connected factorisation condition.

    Theorem 3.21: The double category of lenses Lens is isomorphic to Γ(Cof).
    """
    def __init__(self, base_double_cat: Optional[DoubleCategory] = None):
        super().__init__(name="Γ(Cof)")
        self.base_double_cat = base_double_cat or DoubleCategoryCof()

    def lens_to_vertical_arrow(self, lens: DeltaLens) -> Tuple[Cofunctor, DoubleCell, Functor]:
        """
        Maps a DeltaLens (f, ϕ): A -> B to a vertical arrow in Γ(Cof):
        A cell α in Cof:
               f
           A -----> B
           |        |
         ϕ |   α    | 1^c_B
           v        v
           B -----> B
              1_B
        """
        top = lens.forward
        bottom = identity_functor(lens.target)
        left = lens.cofunctor
        right = identity_cofunctor(lens.target)

        # In Cof, cell condition is:
        # 1_B(top(ϕ(a, u))) = right.lift(top(a), bottom(u))
        # f(ϕ(a, u)) = 1^c_B.lift(fa, u) = u.
        # This is exactly Lens Axiom (L1)!
        cell = DoubleCell(
            name=f"Cell_L1({lens.name})",
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            payload=lens,
        )
        return (left, cell, top)

    def vertical_arrow_to_lens(self, left: Cofunctor, cell: DoubleCell, top: Functor) -> DeltaLens:
        """
        Maps a vertical arrow (f, α, f') in Γ(Cof) back to a DeltaLens.
        By Theorem 3.21, f' = f and the cell condition guarantees f(ϕ(a, u)) = u.
        """
        return DeltaLens(
            name=f"Lens({top.name})",
            forward_functor=top,
            lift_fn=left.lift,
        )

    def compose_vertical_arrows(
        self,
        v2: Tuple[Cofunctor, DoubleCell, Functor],  # B -> C
        v1: Tuple[Cofunctor, DoubleCell, Functor],  # A -> B
    ) -> Tuple[Cofunctor, DoubleCell, Functor]:
        """
        Definition 3.12 (Vertical composition in Γ(D)):
        Vertical composition of (f, α, f') and (g, β, g') is given by the composite cell in D:
             A -----> B -----> C
             |        |        |
           f |   α    | 1_B    | 1_C
             v        v        v
             B -----> B -----> C
             |        |        |
           g |   1    | g      | β
             v        v        v
             C -----> C -----> C
        """
        c1_left, c1_cell, c1_top = v1
        c2_left, c2_cell, c2_top = v2

        # Convert to lenses, compose in Lens, and re-embed in Γ(Cof)
        lens1 = self.vertical_arrow_to_lens(c1_left, c1_cell, c1_top)
        lens2 = self.vertical_arrow_to_lens(c2_left, c2_cell, c2_top)
        comp_lens = lens2.compose(lens1)
        return self.lens_to_vertical_arrow(comp_lens)


class DoubleCategoryLens(DoubleCategory):
    """
    Theorem 3.21: The double category Lens of lenses:
      - Objects: Categories
      - Horizontal morphisms: Functors
      - Vertical morphisms: Delta lenses (f, ϕ)
      - Cells: Squares (top=h, bottom=k, left=(f,ϕ), right=(g,γ)) such that:
          1) g ∘ h = k ∘ f (commutes horizontally)
          2) h_1(ϕ(a, u)) = γ(h_0(a), k_1(u)) for all a ∈ A_0, u: fa -> b
    """
    def __init__(self):
        super().__init__(name="Lens")
        self.completion = RightConnectedCompletion()

    def make_cell(
        self,
        name: str,
        top: Functor,
        bottom: Functor,
        left: DeltaLens,
        right: DeltaLens,
    ) -> DoubleCell:
        cell = DoubleCell(name=name, top=top, bottom=bottom, left=left, right=right)
        valid, msg = self.validate_cell(cell)
        if not valid:
            raise ValueError(f"Cell {name} violates Lens compatibility: {msg}")
        return cell

    def validate_cell(self, cell: DoubleCell) -> Tuple[bool, Optional[str]]:
        h = cell.top
        k = cell.bottom
        left: DeltaLens = cell.left
        right: DeltaLens = cell.right

        # Condition 1: g ∘ h = k ∘ f on objects and morphisms
        for a in left.source.objects:
            gh_a = right.get(h.on_object(a))
            kf_a = k.on_object(left.get(a))
            if gh_a != kf_a:
                return False, f"Horizontal functor mismatch at object {a}: g(h(a))={gh_a} != k(f(a))={kf_a}"

        for m in left.source.morphisms:
            gh_m = right.forward.on_morphism(h.on_morphism(m))
            kf_m = k.on_morphism(left.forward.on_morphism(m))
            if gh_m != kf_m:
                return False, f"Horizontal functor mismatch at morphism {m}: g(h(m))={gh_m} != k(f(m))={kf_m}"

        # Condition 2: h(ϕ(a, u)) = γ(ha, ku)
        for a in left.source.objects:
            fa = left.get(a)
            ha = h.on_object(a)
            for u in left.target.morphisms:
                if left.target.dom(u) == fa:
                    phi_a_u = left.lift(a, u)
                    h_phi = h.on_morphism(phi_a_u)
                    ku = k.on_morphism(u)
                    gamma_ha_ku = right.lift(ha, ku)
                    if h_phi != gamma_ha_ku:
                        return False, f"Lifting mismatch at (a={a}, u={u}): h(ϕ(a,u))={h_phi} != γ(ha, ku)={gamma_ha_ku}"

        return True, None
