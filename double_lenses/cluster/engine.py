"""
Distributed Lens Runtime Engine.
Coordinates multi-device and multi-node execution for forward and adjoint passes.
"""

import time
from typing import Any, Union

import numpy as np

from double_lenses.autodiff.loss import CrossEntropyLossLens
from double_lenses.autodiff.param_lens import LensContext, ParameterizedLens
from double_lenses.autodiff.tensor import Tensor
from double_lenses.cluster.fabric import CommunicationFabric
from double_lenses.cluster.offloading import MemoryStager
from double_lenses.cluster.sharding import ExpertParallel2Cell
from double_lenses.cluster.topology import ClusterTopology, DeviceAddress
from double_lenses.models.deepseek import DeepSeekModelLens
from double_lenses.models.mixtral import MixtralModelLens


class DistributedLensRuntime:
    """
    Cluster Execution Runtime for Lenses:
      - Manages cluster topology and communication fabric
      - Partitions and stages models across heterogeneous GPUs and CPUs
      - Executes coordinated forward and adjoint passes
    """

    def __init__(self, topology: ClusterTopology, fabric: CommunicationFabric | None = None):
        self.topology = topology
        self.fabric = fabric or CommunicationFabric(topology)
        self.ingress_device = topology.all_devices()[0]
        self.expert_cells: dict[str, ExpertParallel2Cell] = {}
        self.memory_stagers: dict[str, MemoryStager] = {}
        self.stats: dict[str, Any] = {
            "forward_passes": 0,
            "adjoint_passes": 0,
            "forward_time_ms": 0.0,
            "adjoint_time_ms": 0.0,
        }

    def start(self) -> None:
        self.fabric.start()

    def stop(self) -> None:
        self.fabric.stop()

    def stage_mixtral(
        self,
        model: MixtralModelLens,
        expert_distribution_mode: str = "heterogeneous",
    ) -> None:
        """
        Stages Mixtral model across cluster devices:
        Distributes experts of each block across available GPUs and CPUs.
        """
        all_devs = self.topology.all_devices()
        num_devs = len(all_devs)

        for layer_idx, layer in enumerate(model.layers):
            moe = layer.moe
            num_experts = moe.num_experts

            # Allocate experts round-robin or GPU-prioritized
            placements: dict[int, DeviceAddress] = {}
            for e_id in range(num_experts):
                dev = all_devs[e_id % num_devs]
                placements[e_id] = dev

            ep_cell = ExpertParallel2Cell(
                name=f"layer_{layer_idx}.moe_ep",
                dim=moe.dim,
                hidden_dim=moe.hidden_dim,
                num_experts=num_experts,
                top_k=moe.top_k,
                expert_placements=placements,
                fabric=self.fabric,
                ingress_device=self.ingress_device,
            )
            self.expert_cells[f"mixtral_layer_{layer_idx}"] = ep_cell

    def stage_deepseek(
        self,
        model: DeepSeekModelLens,
        expert_distribution_mode: str = "heterogeneous",
    ) -> None:
        """
        Stages DeepSeek model across cluster devices:
        Stages fine-grained routed experts across GPUs and CPUs while keeping
        shared experts on primary devices.
        """
        all_devs = self.topology.all_devices()
        num_devs = len(all_devs)

        for layer_idx, layer in enumerate(model.layers):
            moe = layer.moe
            num_routed = moe.num_routed

            placements: dict[int, DeviceAddress] = {}
            for e_id in range(num_routed):
                dev = all_devs[e_id % num_devs]
                placements[e_id] = dev

            ep_cell = ExpertParallel2Cell(
                name=f"layer_{layer_idx}.deepseek_moe_ep",
                dim=moe.dim,
                hidden_dim=moe.routed_hidden_dim,
                num_experts=num_routed,
                top_k=moe.top_k,
                expert_placements=placements,
                fabric=self.fabric,
                ingress_device=self.ingress_device,
            )
            self.expert_cells[f"deepseek_layer_{layer_idx}"] = ep_cell

    def run_forward(
        self,
        model: ParameterizedLens,
        ctx: LensContext,
        tokens: Union[Tensor, np.ndarray],
    ) -> Tensor:
        """Executes distributed forward pass."""
        t0 = time.perf_counter()
        out = model.forward(ctx, tokens)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self.stats["forward_passes"] += 1
        self.stats["forward_time_ms"] += elapsed
        return out

    def run_adjoint(
        self,
        model: ParameterizedLens,
        ctx: LensContext,
        grad_output: Tensor,
    ) -> Any:
        """Executes distributed adjoint pass."""
        t0 = time.perf_counter()
        res = model.adjoint(ctx, grad_output)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self.stats["adjoint_passes"] += 1
        self.stats["adjoint_time_ms"] += elapsed
        return res

    def run_step(
        self,
        model: ParameterizedLens,
        loss_lens: CrossEntropyLossLens,
        tokens: Union[Tensor, np.ndarray],
        targets: Union[Tensor, np.ndarray],
    ) -> tuple[float, dict[str, Any]]:
        """
        Executes a complete forward + adjoint training/inference step:
          1. Forward pass through model lens
          2. Forward pass through loss lens
          3. Adjoint pass through loss lens to generate cotangent seed
          4. Adjoint pass through model lens to accumulate parameter gradients across cluster
        """
        ctx_model = LensContext()
        ctx_loss = LensContext()

        # 1. Forward
        logits = self.run_forward(model, ctx_model, tokens)

        # 2. Loss
        loss_val = loss_lens.forward(ctx_loss, logits, targets)

        # 3. Adjoint Seed
        grad_logits = loss_lens.adjoint(ctx_loss)

        # 4. Model Adjoint (Backprop)
        self.run_adjoint(model, ctx_model, grad_logits)

        return loss_val.item(), self.stats
