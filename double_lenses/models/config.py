"""
Configuration dataclasses for Mixtral and DeepSeek architectures.
Includes both production configurations and lightweight mini test configurations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MixtralConfig:
    """
    Configuration for Mixtral 8x7B / 8x22B architecture.
    Features:
      - Grouped-Query Attention (GQA)
      - Rotary Position Embeddings (RoPE)
      - Sparse Mixture-of-Experts (SMoE) with Top-2 routing across 8 SwiGLU experts
    """
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    hidden_dim: int = 14336
    num_experts: int = 8
    top_k: int = 2
    vocab_size: int = 32000
    max_seq_len: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0

    @classmethod
    def mixtral_mini(cls) -> 'MixtralConfig':
        """Lightweight configuration for rapid unit tests, verification, and staging."""
        return cls(
            dim=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            head_dim=32,
            hidden_dim=256,
            num_experts=4,
            top_k=2,
            vocab_size=1000,
            max_seq_len=256,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
        )

    @classmethod
    def mixtral_8x7b(cls) -> 'MixtralConfig':
        """Full Mixtral 8x7B configuration."""
        return cls()


@dataclass
class DeepSeekConfig:
    """
    Configuration for DeepSeek-V2 / DeepSeek-V3 architecture.
    Features:
      - Multi-Head Latent Attention (MLA): Low-rank KV compression & decoupled RoPE
      - DeepSeekMoE: Fine-grained routed experts + isolated shared experts
    """
    dim: int = 2048
    n_layers: int = 28
    n_heads: int = 16
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    num_routed_experts: int = 64
    num_shared_experts: int = 2
    top_k: int = 6
    routed_hidden_dim: int = 1408
    shared_hidden_dim: int = 2816
    vocab_size: int = 102400
    max_seq_len: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    @classmethod
    def deepseek_mini(cls) -> 'DeepSeekConfig':
        """Lightweight configuration for rapid unit tests and cluster staging."""
        return cls(
            dim=128,
            n_layers=2,
            n_heads=4,
            kv_lora_rank=32,
            q_lora_rank=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            num_routed_experts=4,
            num_shared_experts=1,
            top_k=2,
            routed_hidden_dim=128,
            shared_hidden_dim=128,
            vocab_size=1000,
            max_seq_len=256,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
        )

    @classmethod
    def deepseek_v3(cls) -> 'DeepSeekConfig':
        """Production DeepSeek-V3 configuration."""
        return cls(
            dim=7168,
            n_layers=61,
            n_heads=128,
            kv_lora_rank=512,
            q_lora_rank=1536,
            qk_rope_head_dim=64,
            v_head_dim=128,
            num_routed_experts=256,
            num_shared_experts=1,
            top_k=8,
            routed_hidden_dim=2048,
            shared_hidden_dim=2048,
            vocab_size=129280,
            max_seq_len=4096,
        )
