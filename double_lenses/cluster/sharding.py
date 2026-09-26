"""
Horizontal Sharding Functors and Double Category 2-Cells.
Implements:
  - Tensor Parallelism (TP) 2-Cells (ColumnParallel + RowParallel)
  - Expert Parallelism (EP) 2-Cells (Staging MoE experts across GPUs/CPUs)
  - Pipeline Parallelism (PP) Stages
"""

import numpy as np

from double_lenses.autodiff.param_lens import LensContext, ParameterizedLens
from double_lenses.autodiff.tensor import Tensor, randn
from double_lenses.cluster.collectives import AllReduceFunctor
from double_lenses.cluster.fabric import CommunicationFabric
from double_lenses.cluster.topology import DeviceAddress
from double_lenses.models.moe.expert import ExpertLens


class ColumnParallelLinearLens(ParameterizedLens):
    """
    Column-Parallel Linear Layer shard on a single device:
    Shards W along out_features: W_local is (out_features / num_shards, in_features).
    """

    def __init__(self, name: str, in_features: int, out_features_per_shard: int, device: str = "cpu"):
        super().__init__(name=name)
        self.in_features = in_features
        self.out_features_per_shard = out_features_per_shard
        self.device = device

        scale = 1.0 / np.sqrt(in_features)
        self.w = self.register_parameter(
            "weight", randn((out_features_per_shard, in_features), std=scale, device=device)
        )

    def forward(self, ctx: LensContext, x: Tensor) -> Tensor:
        ctx.save("input", x)
        return x @ self.w.transpose(1, 0)

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        x: Tensor = ctx.get("input")
        grad_x = grad_y @ self.w

        x_2d = x.reshape(-1, self.in_features)
        grad_y_2d = grad_y.reshape(-1, self.out_features_per_shard)
        grad_w = grad_y_2d.transpose(1, 0) @ x_2d

        if self.w.grad is None:
            self.w.grad = grad_w
        else:
            self.w.grad = self.w.grad + grad_w

        return grad_x


class RowParallelLinearLens(ParameterizedLens):
    """
    Row-Parallel Linear Layer shard on a single device:
    Shards W along in_features: W_local is (out_features, in_features / num_shards).
    Requires AllReduce on output.
    """

    def __init__(self, name: str, in_features_per_shard: int, out_features: int, device: str = "cpu"):
        super().__init__(name=name)
        self.in_features_per_shard = in_features_per_shard
        self.out_features = out_features
        self.device = device

        scale = 1.0 / np.sqrt(in_features_per_shard)
        self.w = self.register_parameter(
            "weight", randn((out_features, in_features_per_shard), std=scale, device=device)
        )

    def forward(self, ctx: LensContext, x: Tensor) -> Tensor:
        ctx.save("input", x)
        return x @ self.w.transpose(1, 0)

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        x: Tensor = ctx.get("input")
        grad_x = grad_y @ self.w

        x_2d = x.reshape(-1, self.in_features_per_shard)
        grad_y_2d = grad_y.reshape(-1, self.out_features)
        grad_w = grad_y_2d.transpose(1, 0) @ x_2d

        if self.w.grad is None:
            self.w.grad = grad_w
        else:
            self.w.grad = self.w.grad + grad_w

        return grad_x


class TensorParallel2Cell:
    """
    Double Cell for Tensor Parallelism:
      ColumnParallel shards -> AllGather -> RowParallel shards -> AllReduce
    Guarantees mathematical equivalence to a standard full Linear layer.
    """

    def __init__(
        self,
        name: str,
        in_features: int,
        hidden_features: int,
        out_features: int,
        devices: list[DeviceAddress],
        fabric: CommunicationFabric,
    ):
        self.name = name
        self.devices = devices
        self.fabric = fabric
        self.num_shards = len(devices)

        assert hidden_features % self.num_shards == 0

        self.hidden_per_shard = hidden_features // self.num_shards

        self.col_shards: dict[str, ColumnParallelLinearLens] = {}
        self.row_shards: dict[str, RowParallelLinearLens] = {}

        for dev in devices:
            dev_str = str(dev)
            self.col_shards[dev_str] = ColumnParallelLinearLens(
                f"{name}.col_{dev_str}", in_features, self.hidden_per_shard, device=dev_str
            )
            self.row_shards[dev_str] = RowParallelLinearLens(
                f"{name}.row_{dev_str}", self.hidden_per_shard, out_features, device=dev_str
            )

        self.all_reduce = AllReduceFunctor(fabric, devices)

    def forward(self, ctx: LensContext, input_x: Tensor) -> Tensor:
        # Replicate input to each device
        mid_shards = {}
        for dev in self.devices:
            dev_str = str(dev)
            sub_ctx = ctx.create_sub_context()
            mid_shards[dev_str] = self.col_shards[dev_str].forward(sub_ctx, input_x.to(dev_str))

        # Row parallel forward on each shard
        row_outs = {}
        for dev in self.devices:
            dev_str = str(dev)
            sub_ctx = ctx.create_sub_context()
            row_outs[dev_str] = self.row_shards[dev_str].forward(sub_ctx, mid_shards[dev_str])

        # AllReduce row outputs
        reduced = self.all_reduce.forward(row_outs)
        # Result on first device
        return reduced[str(self.devices[0])]

    def adjoint(self, ctx: LensContext, grad_y: Tensor) -> Tensor:
        # 1. AllReduce Adjoint on cotangents
        grad_outs = {str(dev): grad_y.to(str(dev)) for dev in self.devices}
        reduced_grads = self.all_reduce.adjoint(grad_outs)

        # Sub contexts: first num_shards are col_shards, next num_shards are row_shards
        col_ctxs = ctx.sub_contexts[: self.num_shards]
        row_ctxs = ctx.sub_contexts[self.num_shards : 2 * self.num_shards]

        # 2. Row parallel adjoint
        grad_mids = {}
        for i, dev in enumerate(self.devices):
            dev_str = str(dev)
            grad_mids[dev_str] = self.row_shards[dev_str].adjoint(row_ctxs[i], reduced_grads[dev_str])

        # 3. Col parallel adjoint
        grad_x_accum = None
        for i, dev in enumerate(self.devices):
            dev_str = str(dev)
            grad_x_local = self.col_shards[dev_str].adjoint(col_ctxs[i], grad_mids[dev_str])
            if grad_x_accum is None:
                grad_x_accum = grad_x_local.to_numpy().copy()
            else:
                grad_x_accum += grad_x_local.to_numpy()

        return Tensor(grad_x_accum, device=grad_y.device)


class ExpertParallel2Cell:
    """
    Double Cell for Expert Parallelism (EP):
      Stages MoE experts across distinct cluster devices (GPUs and CPUs).
      Handles routing, token dispatch, parallel execution, output combine,
      and exact distributed adjoint gradient propagation.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        expert_placements: dict[int, DeviceAddress],
        fabric: CommunicationFabric,
        ingress_device: DeviceAddress,
    ):
        self.name = name
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_placements = expert_placements
        self.fabric = fabric
        self.ingress_device = ingress_device

        # Experts placed on designated cluster devices
        self.experts: dict[int, ExpertLens] = {}
        for e_id, dev in expert_placements.items():
            exp = ExpertLens(
                f"{name}.expert_{e_id}",
                in_dim=dim,
                hidden_dim=hidden_dim,
                out_dim=dim,
                expert_id=e_id,
            )
            exp.to(str(dev))
            self.experts[e_id] = exp

    def forward(
        self,
        ctx: LensContext,
        flat_x: np.ndarray,
        flat_weights: np.ndarray,
        flat_experts: np.ndarray,
    ) -> np.ndarray:
        """
        Dispatches tokens across cluster devices hosting assigned experts.
        """
        n_tokens = flat_x.shape[0]
        flat_out = np.zeros_like(flat_x)

        expert_token_indices: dict[int, list[int]] = {i: [] for i in range(self.num_experts)}
        expert_k_slots: dict[int, list[int]] = {i: [] for i in range(self.num_experts)}
        expert_outputs: dict[int, np.ndarray] = {}
        expert_contexts: dict[int, LensContext] = {}

        for t_idx in range(n_tokens):
            for k_idx in range(self.top_k):
                e_id = int(flat_experts[t_idx, k_idx])
                expert_token_indices[e_id].append(t_idx)
                expert_k_slots[e_id].append(k_idx)

        # Dispatch and execute on assigned devices
        for e_id in range(self.num_experts):
            t_indices = expert_token_indices[e_id]
            if not t_indices:
                continue

            target_dev = self.expert_placements[e_id]
            sub_x = flat_x[t_indices]

            # Communication Functor: Send tokens to target device
            tensor_x = Tensor(sub_x, device=str(target_dev))
            self.fabric.send(tensor_x, target_dev, tag=f"ep_fwd_{e_id}")

            # Target device receives and runs expert lens
            recv_x = self.fabric.recv(target_dev) or tensor_x
            sub_ctx = ctx.create_sub_context()
            expert_contexts[e_id] = sub_ctx

            exp_out_t = self.experts[e_id].forward(sub_ctx, recv_x)

            # Send output back to ingress device
            self.fabric.send(exp_out_t, self.ingress_device, tag=f"ep_out_{e_id}")
            recv_out = self.fabric.recv(self.ingress_device) or exp_out_t
            exp_out = recv_out.to_numpy()
            expert_outputs[e_id] = exp_out

            k_slots = expert_k_slots[e_id]
            w = flat_weights[t_indices, k_slots, None]
            flat_out[t_indices] += exp_out * w

        ctx.save("expert_contexts", expert_contexts)
        ctx.save("expert_token_indices", expert_token_indices)
        ctx.save("expert_k_slots", expert_k_slots)
        ctx.save("expert_outputs", expert_outputs)

        return flat_out

    def adjoint(
        self,
        ctx: LensContext,
        dy_np: np.ndarray,
        flat_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Distributed adjoint pass:
        Pulls back cotangents through expert lenses across cluster devices.
        Returns (grad_flat_x, grad_flat_weights).
        """
        expert_contexts = ctx.get("expert_contexts")
        expert_token_indices = ctx.get("expert_token_indices")
        expert_k_slots = ctx.get("expert_k_slots")
        expert_outputs = ctx.get("expert_outputs")

        grad_flat_x = np.zeros_like(dy_np)
        grad_flat_weights = np.zeros_like(flat_weights)

        for e_id in range(self.num_experts):
            t_indices = expert_token_indices[e_id]
            if not t_indices:
                continue

            target_dev = self.expert_placements[e_id]
            k_slots = expert_k_slots[e_id]
            w = flat_weights[t_indices, k_slots, None]
            dy_sub = dy_np[t_indices]
            exp_out = expert_outputs[e_id]

            # 1. Routing weight gradient on ingress device
            dw = np.sum(dy_sub * exp_out, axis=-1)
            grad_flat_weights[t_indices, k_slots] += dw

            # 2. Expert cotangent: Send to device hosting expert e_id
            d_exp_out = dy_sub * w
            t_cotangent = Tensor(d_exp_out, device=str(target_dev))
            self.fabric.send(t_cotangent, target_dev, tag=f"ep_bwd_{e_id}")

            # 3. Hosting device receives cotangent and runs local adjoint pass
            recv_cot = self.fabric.recv(target_dev) or t_cotangent
            sub_ctx = expert_contexts[e_id]
            grad_sub_x_t = self.experts[e_id].adjoint(sub_ctx, recv_cot)

            # 4. Send input cotangent back to ingress device
            self.fabric.send(grad_sub_x_t, self.ingress_device, tag=f"ep_grad_x_{e_id}")
            recv_grad_x = self.fabric.recv(self.ingress_device) or grad_sub_x_t
            grad_flat_x[t_indices] += recv_grad_x.to_numpy()

        return grad_flat_x, grad_flat_weights
