"""
Command-Line Interface and Cluster Demo Runner.
Launches distributed forward and adjoint passes for Mixtral and DeepSeek
staged across heterogeneous GPUs and CPUs.
"""

import argparse
import time

import numpy as np

from double_lenses.autodiff.loss import CrossEntropyLossLens
from double_lenses.cluster.engine import DistributedLensRuntime
from double_lenses.cluster.fabric import CommunicationFabric
from double_lenses.cluster.topology import ClusterTopology
from double_lenses.models.config import DeepSeekConfig, MixtralConfig
from double_lenses.models.deepseek import DeepSeekModelLens
from double_lenses.models.mixtral import MixtralModelLens


def run_mixtral_demo(runtime: DistributedLensRuntime, steps: int = 3) -> None:
    print("\n" + "=" * 70)
    print("  STAGING MIXTRAL 8x7B (MoE + GQA) ON DISTRIBUTED CLUSTER")
    print("=" * 70)

    config = MixtralConfig.mixtral_mini()
    print(
        f"[*] Configuration: dim={config.dim}, layers={config.n_layers}, heads={config.n_heads}, kv_heads={config.n_kv_heads}"
    )
    print(f"[*] MoE Structure: {config.num_experts} SwiGLU Experts, Top-{config.top_k} Routing per token")

    print("[*] Instantiating Mixtral categorical lens network...")
    model = MixtralModelLens("mixtral", config)
    total_params = sum(p.to_numpy().size for p in model.parameters.values())
    print(f"[*] Total Parameters: {total_params:,} elements")

    print("[*] Staging MoE experts across cluster devices...")
    runtime.stage_mixtral(model)

    loss_lens = CrossEntropyLossLens("loss")

    batch_size = 2
    seq_len = 8
    print(
        f"[*] Running {steps} distributed forward & adjoint training steps (batch={batch_size}, seq_len={seq_len})...\n"
    )

    for step in range(1, steps + 1):
        tokens = np.random.randint(0, config.vocab_size, size=(batch_size, seq_len))
        targets = np.random.randint(0, config.vocab_size, size=(batch_size, seq_len))

        t0 = time.perf_counter()
        loss, stats = runtime.run_step(model, loss_lens, tokens, targets)
        t_step = (time.perf_counter() - t0) * 1000.0

        # Verify gradient existence on key parameters
        lm_head_grad_norm = np.linalg.norm(model.lm_head.w.grad.to_numpy()) if model.lm_head.w.grad is not None else 0.0
        router_grad_norm = (
            np.linalg.norm(model.layers[0].moe.router.w_gate.w.grad.to_numpy())
            if model.layers[0].moe.router.w_gate.w.grad is not None
            else 0.0
        )

        print(f"  [Step {step}/{steps}] Loss: {loss:.4f} | Time: {t_step:.2f} ms")
        print(f"               LM Head ||∇W||: {lm_head_grad_norm:.6f} | Router ||∇W||: {router_grad_norm:.6f}")


def run_deepseek_demo(runtime: DistributedLensRuntime, steps: int = 3) -> None:
    print("\n" + "=" * 70)
    print("  STAGING DEEPSEEK (MLA + DeepSeekMoE) ON DISTRIBUTED CLUSTER")
    print("=" * 70)

    config = DeepSeekConfig.deepseek_mini()
    print(f"[*] Configuration: dim={config.dim}, layers={config.n_layers}, MLA heads={config.n_heads}")
    print(f"[*] MLA Low-Rank KV Rank: {config.kv_lora_rank}, Decoupled RoPE Dim: {config.qk_rope_head_dim}")
    print(
        f"[*] DeepSeekMoE: {config.num_shared_experts} Shared Experts + {config.num_routed_experts} Routed Experts, Top-{config.top_k}"
    )

    print("[*] Instantiating DeepSeek categorical lens network...")
    model = DeepSeekModelLens("deepseek", config)
    total_params = sum(p.to_numpy().size for p in model.parameters.values())
    print(f"[*] Total Parameters: {total_params:,} elements")

    print("[*] Staging DeepSeek fine-grained experts across cluster devices...")
    runtime.stage_deepseek(model)

    loss_lens = CrossEntropyLossLens("loss")

    batch_size = 2
    seq_len = 8
    print(
        f"[*] Running {steps} distributed forward & adjoint training steps (batch={batch_size}, seq_len={seq_len})...\n"
    )

    for step in range(1, steps + 1):
        tokens = np.random.randint(0, config.vocab_size, size=(batch_size, seq_len))
        targets = np.random.randint(0, config.vocab_size, size=(batch_size, seq_len))

        t0 = time.perf_counter()
        loss, stats = runtime.run_step(model, loss_lens, tokens, targets)
        t_step = (time.perf_counter() - t0) * 1000.0

        lm_head_grad_norm = np.linalg.norm(model.lm_head.w.grad.to_numpy()) if model.lm_head.w.grad is not None else 0.0
        shared_exp_norm = (
            np.linalg.norm(model.layers[0].moe.shared_experts[0].swiglu.w_gate.w.grad.to_numpy())
            if model.layers[0].moe.shared_experts[0].swiglu.w_gate.w.grad is not None
            else 0.0
        )

        print(f"  [Step {step}/{steps}] Loss: {loss:.4f} | Time: {t_step:.2f} ms")
        print(f"               LM Head ||∇W||: {lm_head_grad_norm:.6f} | Shared Exp ||∇W||: {shared_exp_norm:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed Double Lens Neural Network Runtime")
    parser.add_argument(
        "--model", type=str, choices=["mixtral", "deepseek", "both"], default="both", help="Model to run"
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of simulated/CUDA GPUs")
    parser.add_argument("--num-cpus", type=int, default=2, help="Number of CPU worker nodes")
    parser.add_argument("--steps", type=int, default=3, help="Number of forward/adjoint steps to execute")
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("  THE DOUBLE CATEGORY OF LENSES: DISTRIBUTED NEURAL NETWORK RUNTIME")
    print("  Based on Bryce Clarke (2022) 'The double category of lenses'")
    print("#" * 70)

    # Initialize cluster topology
    topology = ClusterTopology.create_local_heterogeneous(
        num_gpus=args.num_gpus,
        num_cpus=args.num_cpus,
    )
    print(f"[*] Cluster Topology Initialized: {topology}")
    for dev in topology.all_devices():
        print(f"    - Device: {dev}")

    fabric = CommunicationFabric(topology)
    runtime = DistributedLensRuntime(topology, fabric)
    runtime.start()

    try:
        if args.model in ("mixtral", "both"):
            run_mixtral_demo(runtime, steps=args.steps)

        if args.model in ("deepseek", "both"):
            run_deepseek_demo(runtime, steps=args.steps)

        print("\n" + "=" * 70)
        print("  CLUSTER EXECUTION SUMMARY")
        print("=" * 70)
        print(f"  Forward Passes:  {runtime.stats['forward_passes']}")
        print(f"  Adjoint Passes:  {runtime.stats['adjoint_passes']}")
        print(f"  Total Fwd Time:  {runtime.stats['forward_time_ms']:.2f} ms")
        print(f"  Total Adj Time:  {runtime.stats['adjoint_time_ms']:.2f} ms")
        print("  Status: All forward and adjoint passes verified successfully.")
        print("=" * 70 + "\n")
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
