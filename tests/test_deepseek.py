"""
Unit and Integration Tests for DeepSeek Architecture Lenses (MLA, DeepSeekMoE, Model).
"""

import numpy as np
from double_lenses.autodiff.tensor import Tensor, randn
from double_lenses.autodiff.param_lens import LensContext
from double_lenses.autodiff.loss import CrossEntropyLossLens
from double_lenses.models.config import DeepSeekConfig
from double_lenses.models.attention.mla import MultiHeadLatentAttentionLens
from double_lenses.models.moe.deepseek_moe import DeepSeekMoELens
from double_lenses.models.deepseek import DeepSeekTransformerBlockLens, DeepSeekModelLens


def test_mla_forward_and_adjoint():
    config = DeepSeekConfig.deepseek_mini()
    mla = MultiHeadLatentAttentionLens("mla", config)

    batch_size = 2
    seq_len = 4
    x = randn((batch_size, seq_len, config.dim))

    ctx = LensContext()
    out = mla.forward(ctx, x)
    assert out.shape == (batch_size, seq_len, config.dim)

    # Adjoint pass
    dy = randn(out.shape)
    dx = mla.adjoint(ctx, dy)
    assert dx.shape == x.shape

    # Check key projection gradients
    assert mla.w_dkv.w.grad is not None
    assert mla.w_dq.w.grad is not None
    assert mla.out_proj.w.grad is not None


def test_deepseek_moe_forward_and_adjoint():
    config = DeepSeekConfig.deepseek_mini()
    moe = DeepSeekMoELens("deepseek_moe", config)

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

    # Check router, shared expert, and routed expert gradients
    assert moe.router.w_gate.w.grad is not None
    assert moe.shared_experts[0].swiglu.w_gate.w.grad is not None
    active_routed = [e for e in moe.routed_experts if e.swiglu.w_gate.w.grad is not None]
    assert len(active_routed) > 0


def test_deepseek_model_end_to_end():
    config = DeepSeekConfig.deepseek_mini()
    model = DeepSeekModelLens("deepseek", config)
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
