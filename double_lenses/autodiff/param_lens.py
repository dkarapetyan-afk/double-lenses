"""
Parameterized Lenses Para(Lens) and Contexts for Adjoint Autodiff.
Implements the categorical chain rule and gradient accumulation.
"""

from typing import Any

from double_lenses.autodiff.tensor import Tensor


class LensContext:
    """
    Context object that stores intermediate forward values needed for the adjoint lift.
    Corresponds to the morphism in the Category of Chosen Lifts Λ(f, ϕ).
    """

    def __init__(self):
        self._saved_tensors: dict[str, Any] = {}
        self.sub_contexts: list[LensContext] = []

    def save(self, key: str, value: Any) -> None:
        self._saved_tensors[key] = value

    def get(self, key: str) -> Any:
        return self._saved_tensors[key]

    def create_sub_context(self) -> "LensContext":
        sub = LensContext()
        self.sub_contexts.append(sub)
        return sub


class ParameterizedLens:
    """
    A Parameterized Lens in Para(Lens):
      - Forward functor: f: P ⊗ X -> Y
      - Adjoint cofunctor: ϕ: (P ⊗ X) ⊗ Y* -> P* ⊗ X*
    Computes both the input cotangent x̄ (pullback) and accumulates parameter gradients ∇w.
    """

    def __init__(self, name: str):
        self.name = name
        self.parameters: dict[str, Tensor] = {}
        self.device: str = "cpu"

    def register_parameter(self, name: str, tensor: Tensor) -> Tensor:
        tensor.requires_grad = True
        self.parameters[name] = tensor
        return tensor

    def zero_grad(self) -> None:
        for p in self.parameters.values():
            p.zero_grad()

    def to(self, device: str) -> "ParameterizedLens":
        self.device = device
        for k, p in self.parameters.items():
            self.parameters[k] = p.to(device)
        return self

    def forward(self, ctx: LensContext, x: Tensor) -> Tensor:
        """
        Forward pass (Functorial evaluation): f(w, x) -> y.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        """
        Adjoint pass (Cofunctorial lifting): ϕ((w, x), ȳ) -> x̄.
        Accumulates ∇w into self.parameters[k].grad and returns input cotangent x̄.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def __call__(self, ctx: LensContext, x: Tensor) -> Tensor:
        return self.forward(ctx, x)

    def compose(self, other: "ParameterizedLens") -> "ComposedLens":
        """Vertical composition: self ∘ other (first other, then self)."""
        return ComposedLens(first=other, second=self)


class ComposedLens(ParameterizedLens):
    """
    Vertical composition of two parameterized lenses: L2 ∘ L1.
    Clarke Eq. 2.1:
      Forward: y = L2(L1(x))
      Adjoint: x̄ = L1.adjoint(L2.adjoint(z̄))
    """

    def __init__(self, first: ParameterizedLens, second: ParameterizedLens):
        super().__init__(name=f"{second.name} ∘ {first.name}")
        self.first = first
        self.second = second

        # Merge parameter references
        for k, v in first.parameters.items():
            self.parameters[f"{first.name}.{k}"] = v
        for k, v in second.parameters.items():
            self.parameters[f"{second.name}.{k}"] = v

    def forward(self, ctx: LensContext, x: Tensor) -> Tensor:
        ctx1 = ctx.create_sub_context()
        ctx2 = ctx.create_sub_context()
        mid = self.first.forward(ctx1, x)
        out = self.second.forward(ctx2, mid)
        return out

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        assert len(ctx.sub_contexts) >= 2, "Context does not contain required sub-contexts"
        ctx1 = ctx.sub_contexts[0]
        ctx2 = ctx.sub_contexts[1]
        grad_mid = self.second.adjoint(ctx2, grad_y)
        grad_x = self.first.adjoint(ctx1, grad_mid)
        return grad_x


class SequentialLens(ParameterizedLens):
    """
    Chains a list of lenses in sequential vertical composition:
    L_N ∘ ... ∘ L_2 ∘ L_1.
    """

    def __init__(self, name: str, lenses: list[ParameterizedLens]):
        super().__init__(name=name)
        self.lenses = lenses
        for idx, lens in enumerate(lenses):
            for k, v in lens.parameters.items():
                self.parameters[f"layer{idx}.{lens.name}.{k}"] = v

    def forward(self, ctx: LensContext, x: Tensor) -> Tensor:
        curr = x
        for lens in self.lenses:
            sub_ctx = ctx.create_sub_context()
            curr = lens.forward(sub_ctx, curr)
        return curr

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        assert len(ctx.sub_contexts) == len(self.lenses), "Context length mismatch in SequentialLens"
        curr_grad = grad_y
        for lens, sub_ctx in zip(reversed(self.lenses), reversed(ctx.sub_contexts)):
            curr_grad = lens.adjoint(sub_ctx, curr_grad)
        return curr_grad
