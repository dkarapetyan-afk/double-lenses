"""
Unit and Integration Tests for Mixtral Architecture Lenses (GQA, MoE, Block, Model).
"""

import numpy as np

from double_lenses.autodiff.loss import CrossEntropyLossLens
from double_lenses.autodiff.param_lens import LensContext
from double_lenses.autodiff.tensor import randn
from double_lenses.models.attention.gqa import GroupedQueryAttentionLens
from double_lenses.models.config import MixtralConfig
from double_lenses.models.mixtral import MixtralModelLens
from double_lenses.models.moe.mixtral_moe import MixtralMoELens


def test_gqa_forward_and_adjoint():
    config = MixtralConfig.mixtral_mini()
    gqa = GroupedQueryAttentionLens("gqa", config)

    batch_size = 2
    seq_len = 4
    x = randn((batch_size, seq_len, config.dim))

    ctx = LensContext()
    out = gqa.forward(ctx, x)
    assert out.shape == (batch_size, seq_len, config.dim)

    # Adjoint pass
    dy = randn(out.shape)
    dx = gqa.adjoint(ctx, dy)
    assert dx.shape == x.shape

    # Gradients accumulated
    assert gqa.q_proj.w.grad is not None
    assert gqa.out_proj.w.grad is not None


def test_mixtral_moe_forward_and_adjoint():
    config = MixtralConfig.mixtral_mini()
    moe = MixtralMoELens("moe", config)

    batch_size = 2
    seq_len = 4
    x = randn((batch_size, seq_len, config.dim))

    ctx = LensContext()
    out = moe.forward(ctx, x)
    assert out.shape == (batch_size, seq_len, config.dim)

    # Adjoint pass
    dy = randn(out.shape)
    dx = moe.adjoint(ctx, dy)
    assert dx.shape == x.shape

    # Check router and expert gradients
    assert moe.router.w_gate.w.grad is not None
    active_experts = [e for e in moe.experts if e.swiglu.w_gate.w.grad is not None]
    assert len(active_experts) > 0


def test_mixtral_model_end_to_end():
    config = MixtralConfig.mixtral_mini()
    model = MixtralModelLens("mixtral", config)
    loss_lens = CrossEntropyLossLens("loss")

    batch_size = 2
    seq_len = 6
    tokens = np.random.randint(0, config.vocab_size, size=(batch_size, seq_len))
    targets = np.random.randint(0, config.vocab_size, size=(batch_size, seq_len))

    ctx_m = LensContext()
    ctx_l = LensContext()

    # 1. Forward
    logits = model.forward(ctx_m, tokens)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)

    # 2. Loss
    loss = loss_lens.forward(ctx_l, logits, targets)
    assert loss.item() > 0.0

    # 3. Adjoint Seed & Backward
    grad_logits = loss_lens.adjoint(ctx_l)
    model.adjoint(ctx_m, grad_logits)

    # 4. Verify gradients
    assert model.lm_head.w.grad is not None
    assert model.tok_embeddings.weight.grad is not None
    assert model.norm.weight.grad is not None
