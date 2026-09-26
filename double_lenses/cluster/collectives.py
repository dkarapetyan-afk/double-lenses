"""
Categorical Collective Functors with Exact Adjoint Duals.
Formalizes distributed communication as horizontal functors in the double category:
  AllGather* = ReduceScatter
  Scatter*   = Gather
  AllReduce* = AllReduce
"""

import numpy as np

from double_lenses.autodiff.tensor import Tensor
from double_lenses.cluster.fabric import CommunicationFabric
from double_lenses.cluster.topology import DeviceAddress


class AllReduceFunctor:
    """
    AllReduce Collective Functor:
      Forward: y_i = Σ_j x_j for all devices i.
      Adjoint Dual: Self-adjoint (AllReduce).
    """

    def __init__(self, fabric: CommunicationFabric, devices: list[DeviceAddress]):
        self.fabric = fabric
        self.devices = devices

    def forward(self, local_tensors: dict[str, Tensor]) -> dict[str, Tensor]:
        """Sums local tensors from each device and distributes the sum back."""
        accum = None
        for dev in self.devices:
            t = local_tensors[str(dev)]
            if accum is None:
                accum = t.to_numpy().copy()
            else:
                accum += t.to_numpy()

        results = {}
        for dev in self.devices:
            results[str(dev)] = Tensor(accum.copy(), device=str(dev))
        return results

    def adjoint(self, grad_outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Adjoint dual of AllReduce is AllReduce."""
        return self.forward(grad_outputs)


class AllGatherFunctor:
    """
    AllGather Collective Functor:
      Forward: Gathers tensor shards along axis and distributes full concatenated tensor.
      Adjoint Dual: ReduceScatter.
    """

    def __init__(self, fabric: CommunicationFabric, devices: list[DeviceAddress], axis: int = 0):
        self.fabric = fabric
        self.devices = devices
        self.axis = axis

    def forward(self, local_shards: dict[str, Tensor]) -> dict[str, Tensor]:
        arrays = [local_shards[str(dev)].to_numpy() for dev in self.devices]
        gathered = np.concatenate(arrays, axis=self.axis)

        results = {}
        for dev in self.devices:
            results[str(dev)] = Tensor(gathered.copy(), device=str(dev))
        return results

    def adjoint(self, grad_outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Adjoint of AllGather is ReduceScatter:
        Sums cotangents across all devices, then slices the shard for each device.
        """
        # Sum cotangents across devices
        accum = None
        for dev in self.devices:
            g = grad_outputs[str(dev)].to_numpy()
            if accum is None:
                accum = g.copy()
            else:
                accum += g

        # Split along axis
        shards = np.split(accum, len(self.devices), axis=self.axis)
        results = {}
        for i, dev in enumerate(self.devices):
            results[str(dev)] = Tensor(shards[i].copy(), device=str(dev))
        return results


class ScatterFunctor:
    """
    Scatter Collective Functor:
      Forward: Splits root tensor along axis and distributes chunks to devices.
      Adjoint Dual: Gather.
    """

    def __init__(self, fabric: CommunicationFabric, root: DeviceAddress, devices: list[DeviceAddress], axis: int = 0):
        self.fabric = fabric
        self.root = root
        self.devices = devices
        self.axis = axis

    def forward(self, root_tensor: Tensor) -> dict[str, Tensor]:
        arr = root_tensor.to_numpy()
        shards = np.split(arr, len(self.devices), axis=self.axis)
        results = {}
        for i, dev in enumerate(self.devices):
            results[str(dev)] = Tensor(shards[i].copy(), device=str(dev))
        return results

    def adjoint(self, grad_shards: dict[str, Tensor]) -> Tensor:
        """Adjoint dual of Scatter is Gather: concatenates worker cotangents."""
        arrays = [grad_shards[str(dev)].to_numpy() for dev in self.devices]
        gathered = np.concatenate(arrays, axis=self.axis)
        return Tensor(gathered, device=str(self.root))


class GatherFunctor:
    """
    Gather Collective Functor:
      Forward: Gathers shards from devices and concatenates onto root.
      Adjoint Dual: Scatter.
    """

    def __init__(self, fabric: CommunicationFabric, root: DeviceAddress, devices: list[DeviceAddress], axis: int = 0):
        self.fabric = fabric
        self.root = root
        self.devices = devices
        self.axis = axis

    def forward(self, local_shards: dict[str, Tensor]) -> Tensor:
        arrays = [local_shards[str(dev)].to_numpy() for dev in self.devices]
        gathered = np.concatenate(arrays, axis=self.axis)
        return Tensor(gathered, device=str(self.root))

    def adjoint(self, grad_root: Tensor) -> dict[str, Tensor]:
        """Adjoint dual of Gather is Scatter."""
        arr = grad_root.to_numpy()
        shards = np.split(arr, len(self.devices), axis=self.axis)
        results = {}
        for i, dev in enumerate(self.devices):
            results[str(dev)] = Tensor(shards[i].copy(), device=str(dev))
        return results
