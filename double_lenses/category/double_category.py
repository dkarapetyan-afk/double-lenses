"""
Double Categories, Double Cells (Squares), and the Flat Double Category Cof.
Based on Bryce Clarke (2022), Chapter 3, Section 3.1 and Appendix A.
"""

from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar
from double_lenses.category.base import Category, Functor, identity_functor
from double_lenses.category.cofunctor import Cofunctor, identity_cofunctor


class DoubleCell:
    """
    A double cell (square) in a double category:
           h
       A -----> C
       |        |
     v |   θ    | w
       v        v
       B -----> D
           k
    Boundary consists of:
      top: h (horizontal morphism A -> C)
      bottom: k (horizontal morphism B -> D)
      left: v (vertical morphism A ⇸ B)
      right: w (vertical morphism C ⇸ D)
    """
    def __init__(
        self,
        name: str,
        top: Functor,
        bottom: Functor,
        left: Any,  # vertical arrow (Cofunctor, DeltaLens, or generic)
        right: Any,  # vertical arrow
        payload: Optional[Any] = None,
    ):
        self.name = name
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.payload = payload

        # Consistency checks on corners
        self.top_left = top.source
        self.top_right = top.target
        self.bottom_left = bottom.source
        self.bottom_right = bottom.target

        assert left.source == self.top_left, "Left arrow source must match top arrow source"
        assert left.target == self.bottom_left, "Left arrow target must match bottom arrow source"
        assert right.source == self.top_right, "Right arrow source must match top arrow target"
        assert right.target == self.bottom_right, "Right arrow target must match bottom arrow target"

    def compose_horizontal(self, other: 'DoubleCell') -> 'DoubleCell':
        """
        Horizontal composition θ_2 ∘_h θ_1:
        self is θ_2 (C -> E), other is θ_1 (A -> C).
        Requires self.left == other.right.
        """
        if self.left != other.right:
            raise ValueError(f"Horizontal cell composition error: self.left ({self.left}) != other.right ({other.right})")

        comp_top = self.top.compose(other.top)
        comp_bottom = self.bottom.compose(other.bottom)
        return DoubleCell(
            name=f"{self.name} ∘_h {other.name}",
            top=comp_top,
            bottom=comp_bottom,
            left=other.left,
            right=self.right,
            payload=(self.payload, other.payload),
        )

    def compose_vertical(self, other: 'DoubleCell') -> 'DoubleCell':
        """
        Vertical composition θ_2 ∘_v θ_1:
        self is θ_2 (B ⇸ X), other is θ_1 (A ⇸ B).
        Requires self.top == other.bottom.
        """
        if self.top != other.bottom:
            raise ValueError(f"Vertical cell composition error: self.top ({self.top}) != other.bottom ({other.bottom})")

        comp_left = self.left.compose(other.left)
        comp_right = self.right.compose(other.right)
        return DoubleCell(
            name=f"{self.name} ∘_v {other.name}",
            top=other.top,
            bottom=self.bottom,
            left=comp_left,
            right=comp_right,
            payload=(self.payload, other.payload),
        )

    def __repr__(self) -> str:
        return (
            f"DoubleCell({self.name}: "
            f"top={self.top.name}, bottom={self.bottom.name}, "
            f"left={self.left.name}, right={self.right.name})"
        )


class DoubleCategory:
    """
    Abstract definition of a strict double category D:
      - Objects: small categories
      - Horizontal morphisms: functors
      - Vertical morphisms: vertical arrows
      - 2-cells: double squares
    """
    def __init__(self, name: str):
        self.name = name

    def horizontal_identity(self, vert: Any) -> DoubleCell:
        """Horizontal identity cell for a vertical morphism v: A ⇸ B."""
        top_id = identity_functor(vert.source)
        bot_id = identity_functor(vert.target)
        return DoubleCell(
            name=f"id_h({vert.name})",
            top=top_id,
            bottom=bot_id,
            left=vert,
            right=vert,
        )

    def vertical_identity(self, horiz: Functor, id_vert_factory: Callable[[Category], Any]) -> DoubleCell:
        """Vertical identity cell for a horizontal functor h: A -> B."""
        left_id = id_vert_factory(horiz.source)
        right_id = id_vert_factory(horiz.target)
        return DoubleCell(
            name=f"id_v({horiz.name})",
            top=horiz,
            bottom=horiz,
            left=left_id,
            right=right_id,
        )

    @staticmethod
    def verify_interchange(
        c11: DoubleCell, c12: DoubleCell,
        c21: DoubleCell, c22: DoubleCell,
    ) -> bool:
        """
        Verifies the Interchange Law:
        (c22 ∘_h c21) ∘_v (c12 ∘_h c11) == (c22 ∘_v c12) ∘_h (c21 ∘_v c11)
        Layout:
          c11   c12
          c21   c22
        """
        # Step 1: horizontal then vertical
        row1 = c12.compose_horizontal(c11)
        row2 = c22.compose_horizontal(c21)
        h_then_v = row2.compose_vertical(row1)

        # Step 2: vertical then horizontal
        col1 = c21.compose_vertical(c11)
        col2 = c22.compose_vertical(c12)
        v_then_h = col2.compose_horizontal(col1)

        # Check boundaries match
        return (
            h_then_v.top.source == v_then_h.top.source
            and h_then_v.top.target == v_then_h.top.target
            and h_then_v.bottom.source == v_then_h.bottom.source
            and h_then_v.bottom.target == v_then_h.bottom.target
            and h_then_v.left.source == v_then_h.left.source
            and h_then_v.left.target == v_then_h.left.target
            and h_then_v.right.source == v_then_h.right.source
            and h_then_v.right.target == v_then_h.right.target
        )


class DoubleCategoryCof(DoubleCategory):
    """
    Definition 3.1: The flat double category Cof of cofunctors:
      - Objects: Small categories
      - Horizontal morphisms: Functors
      - Vertical morphisms: Cofunctors
      - Cells: Squares (top=h, bottom=k, left=(f,ϕ), right=(g,γ)) such that:
          1) g_0 ∘ h_0 = k_0 ∘ f_0
          2) h_1(ϕ(a, u)) = γ(h_0(a), k_1(u)) for all a ∈ A_0, u: fa -> b
    """
    def __init__(self):
        super().__init__(name="Cof")

    def make_cell(
        self,
        name: str,
        top: Functor,
        bottom: Functor,
        left: Cofunctor,
        right: Cofunctor,
    ) -> DoubleCell:
        cell = DoubleCell(name=name, top=top, bottom=bottom, left=left, right=right)
        valid, msg = self.validate_cell(cell)
        if not valid:
            raise ValueError(f"Cell {name} violates Cof compatibility: {msg}")
        return cell

    def validate_cell(self, cell: DoubleCell) -> Tuple[bool, Optional[str]]:
        h = cell.top
        k = cell.bottom
        left: Cofunctor = cell.left
        right: Cofunctor = cell.right

        # Condition 1: g_0 ∘ h_0 = k_0 ∘ f_0 on all objects of A
        for a in left.source.objects:
            gh_a = right.on_object(h.on_object(a))
            kf_a = k.on_object(left.on_object(a))
            if gh_a != kf_a:
                return False, f"Object map mismatch at {a}: g(h(a))={gh_a} != k(f(a))={kf_a}"

        # Condition 2: h_1(ϕ(a, u)) = γ(h_0(a), k_1(u))
        for a in left.source.objects:
            fa = left.on_object(a)
            ha = h.on_object(a)
            for u in left.target.morphisms:
                if left.target.dom(u) == fa:
                    phi_a_u = left.lift(a, u)
                    h_phi = h.on_morphism(phi_a_u)
                    ku = k.on_morphism(u)
                    gamma_ha_ku = right.lift(ha, ku)
                    if h_phi != gamma_ha_ku:
                        return False, f"Lift mismatch at (a={a}, u={u}): h(ϕ(a,u))={h_phi} != γ(ha, ku)={gamma_ha_ku}"

        return True, None
