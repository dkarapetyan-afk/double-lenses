"""
Full Mixtral Transformer Block and End-to-End Language Model Lens.
Composed entirely of categorical lenses with exact analytical forward and adjoint lifting.
"""

from typing import Union

import numpy as np

from double_lenses.autodiff.layers import EmbeddingLens, LinearLens, RMSNormLens
from double_lenses.autodiff.param_lens import LensContext, ParameterizedLens
from double_lenses.autodiff.tensor import Tensor
from double_lenses.models.attention.gqa import GroupedQueryAttentionLens
from double_lenses.models.config import MixtralConfig
from double_lenses.models.moe.mixtral_moe import MixtralMoELens


class MixtralTransformerBlockLens(ParameterizedLens):
    """
    A single Mixtral Transformer block:
      h = x + GQA(RMSNorm_1(x))
      out = h + MoE(RMSNorm_2(h))
    """

    def __init__(self, name: str, config: MixtralConfig):
        super().__init__(name=name)
        self.config = config

        self.norm1 = RMSNormLens(f"{name}.norm1", dim=config.dim, eps=config.rms_norm_eps)
        self.attn = GroupedQueryAttentionLens(f"{name}.attn", config=config)
        self.norm2 = RMSNormLens(f"{name}.norm2", dim=config.dim, eps=config.rms_norm_eps)
        self.moe = MixtralMoELens(f"{name}.moe", config=config)

        # Register parameters
        for sub in [self.norm1, self.attn, self.norm2, self.moe]:
            for k, v in sub.parameters.items():
                self.parameters[f"{sub.name}.{k}"] = v

    def forward(self, ctx: LensContext, x: Tensor, start_pos: int = 0) -> Tensor:
        ctx_norm1 = ctx.create_sub_context()
        ctx_attn = ctx.create_sub_context()
        ctx_norm2 = ctx.create_sub_context()
        ctx_moe = ctx.create_sub_context()

        # Attention sub-block with residual
        norm1_out = self.norm1.forward(ctx_norm1, x)
        attn_out = self.attn.forward(ctx_attn, norm1_out, start_pos=start_pos)
        h = x + attn_out

        # MoE sub-block with residual
        norm2_out = self.norm2.forward(ctx_norm2, h)
        moe_out = self.moe.forward(ctx_moe, norm2_out)
        out = h + moe_out

        return out

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        ctx_norm1, ctx_attn, ctx_norm2, ctx_moe = ctx.sub_contexts[:4]

        # 1. MoE Residual Pullback: out = h + moe_out
        # grad_moe_out = grad_y
        # grad_h_direct = grad_y
        grad_norm2_out = self.moe.adjoint(ctx_moe, grad_y)
        grad_h_norm2 = self.norm2.adjoint(ctx_norm2, grad_norm2_out)
        grad_h = grad_y + grad_h_norm2

        # 2. Attention Residual Pullback: h = x + attn_out
        # grad_attn_out = grad_h
        # grad_x_direct = grad_h
        grad_norm1_out = self.attn.adjoint(ctx_attn, grad_h)
        grad_x_norm1 = self.norm1.adjoint(ctx_norm1, grad_norm1_out)
        grad_x = grad_h + grad_x_norm1

        return grad_x


class MixtralModelLens(ParameterizedLens):
    """
    Complete Mixtral Language Model:
      tokens -> Embedding -> N x TransformerBlocks -> Final RMSNorm -> LM Head -> Logits
    """

    def __init__(self, name: str, config: MixtralConfig):
        super().__init__(name=name)
        self.config = config

        self.tok_embeddings = EmbeddingLens(f"{name}.tok_embeddings", config.vocab_size, config.dim)
        self.layers: list[MixtralTransformerBlockLens] = []
        for i in range(config.n_layers):
            layer = MixtralTransformerBlockLens(f"{name}.layer_{i}", config=config)
            self.layers.append(layer)

        self.norm = RMSNormLens(f"{name}.final_norm", dim=config.dim, eps=config.rms_norm_eps)
        self.lm_head = LinearLens(f"{name}.lm_head", config.dim, config.vocab_size, bias=False)

        # Register parameters
        for k, v in self.tok_embeddings.parameters.items():
            self.parameters[f"tok_embeddings.{k}"] = v
        for i, layer in enumerate(self.layers):
            for k, v in layer.parameters.items():
                self.parameters[f"layer_{i}.{k}"] = v
        for k, v in self.norm.parameters.items():
            self.parameters[f"final_norm.{k}"] = v
        for k, v in self.lm_head.parameters.items():
            self.parameters[f"lm_head.{k}"] = v

    def forward(self, ctx: LensContext, tokens: Union[Tensor, np.ndarray], start_pos: int = 0) -> Tensor:
        ctx_embed = ctx.create_sub_context()
        ctx_norm = ctx.create_sub_context()
        ctx_lm = ctx.create_sub_context()

        # Token embedding
        h = self.tok_embeddings.forward(ctx_embed, tokens)

        # Transformer blocks
        for layer in self.layers:
            sub_ctx = ctx.create_sub_context()
            h = layer.forward(sub_ctx, h, start_pos=start_pos)

        # Final norm and LM head
        norm_h = self.norm.forward(ctx_norm, h)
        logits = self.lm_head.forward(ctx_lm, norm_h)
        return logits

    def adjoint(self, ctx: LensContext, grad_logits: Tensor) -> None:
        ctx_embed = ctx.sub_contexts[0]
        ctx_norm = ctx.sub_contexts[1]
        ctx_lm = ctx.sub_contexts[2]
        layer_contexts = ctx.sub_contexts[3 : 3 + len(self.layers)]

        # 1. Pullback through LM head
        grad_norm_h = self.lm_head.adjoint(ctx_lm, grad_logits)

        # 2. Pullback through final norm
        grad_h = self.norm.adjoint(ctx_norm, grad_norm_h)

        # 3. Pullback through transformer blocks in reverse
        for layer, sub_ctx in zip(reversed(self.layers), reversed(layer_contexts)):
            grad_h = layer.adjoint(sub_ctx, grad_h)

        # 4. Pullback through embedding
        self.tok_embeddings.adjoint(ctx_embed, grad_h)
        return None
