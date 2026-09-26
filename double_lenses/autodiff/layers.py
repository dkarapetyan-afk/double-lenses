"""
Core Neural Network Layers as Parameterized Lenses.
Each layer implements both forward functorial evaluation and
exact adjoint cotangent lifting.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from double_lenses.autodiff.tensor import Tensor, randn, zeros, ones
from double_lenses.autodiff.param_lens import LensContext, ParameterizedLens


class LinearLens(ParameterizedLens):
    """
    Affine transformation layer: y = x W^T + b.
    Forward: Functor f(W, b, x) = x W^T + b
    Adjoint: ϕ((W, b, x), ȳ) = (∇W, ∇b, x̄)
      x̄ = ȳ W
      ∇W = ȳ^T x (aggregated across batch & sequence dimensions)
      ∇b = Σ ȳ
    """
    def __init__(self, name: str, in_features: int, out_features: int, bias: bool = False):
        super().__init__(name=name)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Kaiming uniform / normal init
        scale = 1.0 / np.sqrt(in_features)
        self.w = self.register_parameter("weight", randn((out_features, in_features), std=scale))
        if bias:
            self.b = self.register_parameter("bias", zeros((out_features,)))
        else:
            self.b = None

    def forward(self, ctx: LensContext, x: Tensor) -> Tensor:
        ctx.save("input", x)
        # x is (..., in_features), w is (out_features, in_features)
        y = x @ self.w.transpose(1, 0)
        if self.use_bias and self.b is not None:
            y = y + self.b
        return y

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        x: Tensor = ctx.get("input")

        # Compute input cotangent x̄ = ȳ @ W
        grad_x = grad_y @ self.w

        # Compute parameter gradient ∇W = ȳ^T @ x
        # Handle arbitrary batch/sequence dims
        x_2d = x.reshape(-1, self.in_features)
        grad_y_2d = grad_y.reshape(-1, self.out_features)
        grad_w = grad_y_2d.transpose(1, 0) @ x_2d

        if self.w.grad is None:
            self.w.grad = grad_w
        else:
            self.w.grad = self.w.grad + grad_w

        if self.use_bias and self.b is not None:
            grad_b = grad_y_2d.sum(axis=0)
            if self.b.grad is None:
                self.b.grad = grad_b
            else:
                self.b.grad = self.b.grad + grad_b

        return grad_x


class RMSNormLens(ParameterizedLens):
    """
    Root Mean Square Layer Normalization:
      y = (x / rms(x)) * gamma
      where rms(x) = sqrt(mean(x^2, axis=-1, keepdims=True) + eps)
    """
    def __init__(self, name: str, dim: int, eps: float = 1e-6):
        super().__init__(name=name)
        self.dim = dim
        self.eps = eps
        self.weight = self.register_parameter("weight", ones((dim,)))

    def forward(self, ctx: LensContext, x: Tensor) -> Tensor:
        x_np = x.to_numpy()
        var = np.mean(x_np ** 2, axis=-1, keepdims=True)
        rms = np.sqrt(var + self.eps)
        x_norm = x_np / rms
        y_np = x_norm * self.weight.to_numpy()

        ctx.save("input_np", x_np)
        ctx.save("rms", rms)
        ctx.save("x_norm", x_norm)

        return Tensor(y_np, device=x.device)

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        x_np = ctx.get("input_np")
        rms = ctx.get("rms")
        x_norm = ctx.get("x_norm")
        w_np = self.weight.to_numpy()
        dy_np = grad_y.to_numpy()

        # Gradient w.r.t gamma
        dw_np = np.sum(dy_np * x_norm, axis=tuple(range(dy_np.ndim - 1)))
        grad_w = Tensor(dw_np, device=self.weight.device)
        if self.weight.grad is None:
            self.weight.grad = grad_w
        else:
            self.weight.grad = self.weight.grad + grad_w

        # Gradient w.r.t x (exact adjoint formula)
        # d_x_norm = dy * gamma
        d_xnorm = dy_np * w_np
        # dx = (d_xnorm - x_norm * mean(d_xnorm * x_norm, axis=-1, keepdims=True)) / rms
        dx_np = (d_xnorm - x_norm * np.mean(d_xnorm * x_norm, axis=-1, keepdims=True)) / rms
        return Tensor(dx_np, device=grad_y.device)


class SiLULens(ParameterizedLens):
    """
    SiLU (Swish) Activation: silu(x) = x * sigmoid(x).
    """
    def __init__(self, name: str = "silu"):
        super().__init__(name=name)

    def forward(self, ctx: LensContext, x: Tensor) -> Tensor:
        x_np = x.to_numpy()
        # Numerically stable sigmoid
        sig = 1.0 / (1.0 + np.exp(-np.clip(x_np, -50.0, 50.0)))
        out_np = x_np * sig
        ctx.save("input_np", x_np)
        ctx.save("sig", sig)
        return Tensor(out_np, device=x.device)

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        x_np = ctx.get("input_np")
        sig = ctx.get("sig")
        dy_np = grad_y.to_numpy()
        # silu'(x) = sig + x * sig * (1 - sig) = sig * (1 + x * (1 - sig))
        dsilu = sig * (1.0 + x_np * (1.0 - sig))
        dx_np = dy_np * dsilu
        return Tensor(dx_np, device=grad_y.device)


class SwiGLULens(ParameterizedLens):
    """
    SwiGLU feed-forward network block:
      SwiGLU(x) = (silu(x W_gate^T) * (x W_up^T)) W_down^T
    """
    def __init__(self, name: str, in_dim: int, hidden_dim: int, out_dim: Optional[int] = None):
        super().__init__(name=name)
        out_dim = out_dim or in_dim
        self.w_gate = LinearLens(f"{name}.w_gate", in_dim, hidden_dim, bias=False)
        self.w_up = LinearLens(f"{name}.w_up", in_dim, hidden_dim, bias=False)
        self.w_down = LinearLens(f"{name}.w_down", hidden_dim, out_dim, bias=False)
        self.silu = SiLULens(f"{name}.silu")

        # Register parameters
        for k, v in self.w_gate.parameters.items():
            self.parameters[f"w_gate.{k}"] = v
        for k, v in self.w_up.parameters.items():
            self.parameters[f"w_up.{k}"] = v
        for k, v in self.w_down.parameters.items():
            self.parameters[f"w_down.{k}"] = v

    def forward(self, ctx: LensContext, x: Tensor) -> Tensor:
        ctx_gate = ctx.create_sub_context()
        ctx_up = ctx.create_sub_context()
        ctx_silu = ctx.create_sub_context()
        ctx_down = ctx.create_sub_context()

        gate_act = self.w_gate.forward(ctx_gate, x)
        up_act = self.w_up.forward(ctx_up, x)
        silu_gate = self.silu.forward(ctx_silu, gate_act)

        # Elementwise product
        h = silu_gate * up_act
        ctx.save("silu_gate", silu_gate)
        ctx.save("up_act", up_act)

        out = self.w_down.forward(ctx_down, h)
        return out

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        ctx_gate, ctx_up, ctx_silu, ctx_down = ctx.sub_contexts[:4]
        silu_gate: Tensor = ctx.get("silu_gate")
        up_act: Tensor = ctx.get("up_act")

        # Pullback through w_down
        grad_h = self.w_down.adjoint(ctx_down, grad_y)

        # Pullback through elementwise product: h = silu_gate * up_act
        grad_silu = grad_h * up_act
        grad_up = grad_h * silu_gate

        # Pullback through silu
        grad_gate = self.silu.adjoint(ctx_silu, grad_silu)

        # Pullback through w_gate and w_up and sum
        grad_x_gate = self.w_gate.adjoint(ctx_gate, grad_gate)
        grad_x_up = self.w_up.adjoint(ctx_up, grad_up)

        grad_x = grad_x_gate + grad_x_up
        return grad_x


class SoftmaxLens(ParameterizedLens):
    """
    Numerically stable Softmax layer along specified axis with exact vector-Jacobian adjoint.
    """
    def __init__(self, name: str = "softmax", axis: int = -1):
        super().__init__(name=name)
        self.axis = axis

    def forward(self, ctx: LensContext, x: Tensor) -> Tensor:
        x_np = x.to_numpy()
        max_val = np.max(x_np, axis=self.axis, keepdims=True)
        exp_val = np.exp(x_np - max_val)
        sum_exp = np.sum(exp_val, axis=self.axis, keepdims=True)
        prob = exp_val / sum_exp
        ctx.save("prob", prob)
        return Tensor(prob, device=x.device)

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        p = ctx.get("prob")
        dy = grad_y.to_numpy()
        # J_softmax^T @ dy = p * (dy - sum(p * dy, axis=axis, keepdims=True))
        sum_p_dy = np.sum(p * dy, axis=self.axis, keepdims=True)
        dx = p * (dy - sum_p_dy)
        return Tensor(dx, device=grad_y.device)


class RoPELens(ParameterizedLens):
    """
    Rotary Position Embedding (RoPE) Lens.
    Applies 2D rotation to pairs of features according to token position.
    Adjoint is exact orthogonal transpose (rotation by -θ).
    """
    def __init__(self, name: str, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__(name=name)
        self.head_dim = head_dim
        assert head_dim % 2 == 0, "RoPE head_dim must be even"

        # Precompute frequencies
        dim_indices = np.arange(0, head_dim, 2, dtype=np.float32)
        inv_freq = 1.0 / (theta ** (dim_indices / head_dim))
        positions = np.arange(max_seq_len, dtype=np.float32)
        angles = np.outer(positions, inv_freq)  # (max_seq_len, head_dim // 2)

        self.cos = np.cos(angles)  # (max_seq_len, head_dim // 2)
        self.sin = np.sin(angles)

    def forward(self, ctx: LensContext, x: Tensor, start_pos: int = 0) -> Tensor:
        """
        x is shape (batch, seq_len, n_heads, head_dim) or (batch, seq_len, head_dim)
        """
        x_np = x.to_numpy()
        orig_shape = x_np.shape
        seq_len = orig_shape[1]

        cos_t = self.cos[start_pos:start_pos + seq_len]
        sin_t = self.sin[start_pos:start_pos + seq_len]

        # Reshape to pair consecutive dimensions
        # Split into x1 (even) and x2 (odd)
        x_pairs = x_np.reshape(orig_shape[:-1] + (self.head_dim // 2, 2))
        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]

        # Broadcast cos and sin: (1, seq_len, 1, head_dim // 2) if 4D
        if x_np.ndim == 4:
            cos_b = cos_t[None, :, None, :]
            sin_b = sin_t[None, :, None, :]
        else:
            cos_b = cos_t[None, :, :]
            sin_b = sin_t[None, :, :]

        y1 = x1 * cos_b - x2 * sin_b
        y2 = x1 * sin_b + x2 * cos_b

        y_pairs = np.stack([y1, y2], axis=-1)
        y_np = y_pairs.reshape(orig_shape)

        ctx.save("cos_b", cos_b)
        ctx.save("sin_b", sin_b)
        ctx.save("orig_shape", orig_shape)

        return Tensor(y_np, device=x.device)

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        cos_b = ctx.get("cos_b")
        sin_b = ctx.get("sin_b")
        orig_shape = ctx.get("orig_shape")

        dy_np = grad_y.to_numpy()
        dy_pairs = dy_np.reshape(orig_shape[:-1] + (self.head_dim // 2, 2))
        dy1 = dy_pairs[..., 0]
        dy2 = dy_pairs[..., 1]

        # Inverse rotation (transpose of rotation matrix):
        # [ cos  sin ] [ dy1 ]
        # [-sin  cos ] [ dy2 ]
        dx1 = dy1 * cos_b + dy2 * sin_b
        dx2 = -dy1 * sin_b + dy2 * cos_b

        dx_pairs = np.stack([dx1, dx2], axis=-1)
        dx_np = dx_pairs.reshape(orig_shape)
        return Tensor(dx_np, device=grad_y.device)


class EmbeddingLens(ParameterizedLens):
    """
    Token Embedding Layer.
    Forward: Maps integer token IDs to continuous vectors.
    Adjoint: Accumulates cotangents into the embedding matrix.
    """
    def __init__(self, name: str, vocab_size: int, embed_dim: int):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        scale = 1.0 / np.sqrt(embed_dim)
        self.weight = self.register_parameter("weight", randn((vocab_size, embed_dim), std=scale))

    def forward(self, ctx: LensContext, tokens: Union[Tensor, np.ndarray]) -> Tensor:
        t_np = tokens.to_numpy() if isinstance(tokens, Tensor) else np.array(tokens)
        ctx.save("tokens", t_np)
        w_np = self.weight.to_numpy()
        out_np = w_np[t_np]
        return Tensor(out_np, device=self.weight.device)

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> None:
        t_np = ctx.get("tokens")
        dy_np = grad_y.to_numpy()

        grad_w = np.zeros_like(self.weight.to_numpy())
        np.add.at(grad_w, t_np, dy_np)

        gw_tensor = Tensor(grad_w, device=self.weight.device)
        if self.weight.grad is None:
            self.weight.grad = gw_tensor
        else:
            self.weight.grad = self.weight.grad + gw_tensor
        return None  # No input cotangent for discrete token IDs
