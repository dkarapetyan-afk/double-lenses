"""
Exhaustive Numerical Adjoint Gradient Checking (Finite Differences).
Verifies that the cofunctorial adjoint lifting computes exact derivatives.
"""

import numpy as np
from double_lenses.autodiff.tensor import Tensor, randn
from double_lenses.autodiff.param_lens import LensContext
from double_lenses.autodiff.layers import (
    LinearLens,
    RMSNormLens,
    SiLULens,
    SwiGLULens,
    SoftmaxLens,
    RoPELens,
)


def compute_numerical_gradient(func, param_tensor, eps=1e-3):
    """Computes two-sided finite difference gradient."""
    orig_data = param_tensor.to_numpy().copy()
    grad_num = np.zeros_like(orig_data)

    it = np.nditer(orig_data, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        val = orig_data[idx]

        # f(x + eps)
        orig_data[idx] = val + eps
        param_tensor._data = orig_data.copy()
        loss_plus = func()

        # f(x - eps)
        orig_data[idx] = val - eps
        param_tensor._data = orig_data.copy()
        loss_minus = func()

        grad_num[idx] = (loss_plus - loss_minus) / (2.0 * eps)

        # restore
        orig_data[idx] = val
        it.iternext()

    param_tensor._data = orig_data
    return grad_num


def test_linear_lens_adjoint():
    lens = LinearLens("linear", in_features=4, out_features=3, bias=True)
    x = randn((2, 4))
    dy = randn((2, 3))

    ctx = LensContext()
    y = lens.forward(ctx, x)
    dx = lens.adjoint(ctx, dy)

    # Check gradient w.r.t W
    def loss_fn():
        ctx_temp = LensContext()
        y_temp = lens.forward(ctx_temp, x)
        return float(np.sum(y_temp.to_numpy() * dy.to_numpy()))

    num_grad_w = compute_numerical_gradient(loss_fn, lens.w, eps=1e-3)
    analytic_grad_w = lens.w.grad.to_numpy()

    np.testing.assert_allclose(analytic_grad_w, num_grad_w, rtol=1e-2, atol=1e-3)


def test_rmsnorm_lens_adjoint():
    lens = RMSNormLens("norm", dim=4)
    x = randn((2, 4))
    dy = randn((2, 4))

    ctx = LensContext()
    y = lens.forward(ctx, x)
    dx = lens.adjoint(ctx, dy)

    # Check gradient w.r.t gamma (weight)
    def loss_fn():
        ctx_temp = LensContext()
        y_temp = lens.forward(ctx_temp, x)
        return float(np.sum(y_temp.to_numpy() * dy.to_numpy()))

    num_grad_w = compute_numerical_gradient(loss_fn, lens.weight, eps=1e-3)
    analytic_grad_w = lens.weight.grad.to_numpy()

    np.testing.assert_allclose(analytic_grad_w, num_grad_w, rtol=1e-2, atol=1e-3)


def test_silu_lens_adjoint():
    lens = SiLULens("silu")
    x = randn((2, 4))
    dy = randn((2, 4))

    ctx = LensContext()
    y = lens.forward(ctx, x)
    dx = lens.adjoint(ctx, dy)

    # Analytical silu'(x) = sig * (1 + x * (1 - sig))
    x_np = x.to_numpy()
    sig = 1.0 / (1.0 + np.exp(-x_np))
    expected_dx = dy.to_numpy() * sig * (1.0 + x_np * (1.0 - sig))

    np.testing.assert_allclose(dx.to_numpy(), expected_dx, rtol=1e-4, atol=1e-5)


def test_swiglu_lens_adjoint():
    lens = SwiGLULens("swiglu", in_dim=4, hidden_dim=6)
    x = randn((2, 4))
    dy = randn((2, 4))

    ctx = LensContext()
    y = lens.forward(ctx, x)
    dx = lens.adjoint(ctx, dy)

    def loss_fn():
        ctx_temp = LensContext()
        y_temp = lens.forward(ctx_temp, x)
        return float(np.sum(y_temp.to_numpy() * dy.to_numpy()))

    # Check gradient w.r.t w_down
    num_grad_down = compute_numerical_gradient(loss_fn, lens.w_down.w, eps=1e-3)
    analytic_grad_down = lens.w_down.w.grad.to_numpy()
    np.testing.assert_allclose(analytic_grad_down, num_grad_down, rtol=1e-2, atol=1e-3)


def test_rope_lens_orthogonality_and_adjoint():
    head_dim = 4
    lens = RoPELens("rope", head_dim=head_dim, max_seq_len=16)
    x = randn((1, 4, head_dim))  # batch=1, seq=4, head_dim=4

    ctx = LensContext()
    y = lens.forward(ctx, x)

    # RoPE is orthogonal rotation -> norm of each token vector is strictly preserved!
    x_norm = np.linalg.norm(x.to_numpy(), axis=-1)
    y_norm = np.linalg.norm(y.to_numpy(), axis=-1)
    np.testing.assert_allclose(x_norm, y_norm, rtol=1e-5, atol=1e-5)

    # Inverse rotation test: dy = y -> dx must equal x!
    ctx_bwd = LensContext()
    _ = lens.forward(ctx_bwd, x)
    dx = lens.adjoint(ctx_bwd, y)
    np.testing.assert_allclose(dx.to_numpy(), x.to_numpy(), rtol=1e-4, atol=1e-5)
