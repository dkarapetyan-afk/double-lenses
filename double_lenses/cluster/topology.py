"""
Cluster Topology and Hardware Placement Abstractions.
Represents nodes, devices (CUDA GPUs & CPU worker threads), and network links.
"""

from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class DeviceAddress:
    """
    Unique cluster-wide address for a compute device.
    Examples:
      - DeviceAddress("node0", "cuda", 0) -> "node0:cuda:0"
      - DeviceAddress("node1", "cpu", 0)  -> "node1:cpu:0"
    """

    node_id: str
    device_type: str  # 'cuda' or 'cpu'
    device_id: int = 0

    @classmethod
    def parse(cls, addr_str: str) -> "DeviceAddress":
        parts = addr_str.split(":")
        if len(parts) == 3:
            return cls(node_id=parts[0], device_type=parts[1], device_id=int(parts[2]))
        elif len(parts) == 2:
            return cls(node_id=parts[0], device_type=parts[1], device_id=0)
        elif len(parts) == 1:
            if parts[0].startswith("cuda"):
                dev_id = int(parts[0].split(":")[1]) if ":" in parts[0] else 0
                return cls(node_id="local", device_type="cuda", device_id=dev_id)
            return cls(node_id="local", device_type="cpu", device_id=0)
        raise ValueError(f"Invalid device address string: {addr_str}")

    def __str__(self) -> str:
        return f"{self.node_id}:{self.device_type}:{self.device_id}"


@dataclass
class ClusterNode:
    """A physical or virtual host in the cluster."""

    node_id: str
    host: str = "127.0.0.1"
    port: int = 9000
    devices: list[DeviceAddress] = field(default_factory=list)


class ClusterTopology:
    """
    Manages the cluster hardware graph, device placements, and interconnect links.
    """

    def __init__(self):
        self.nodes: dict[str, ClusterNode] = {}
        self.devices: dict[str, DeviceAddress] = {}

    def add_node(self, node: ClusterNode) -> None:
        self.nodes[node.node_id] = node
        for dev in node.devices:
            self.devices[str(dev)] = dev

    def get_device(self, addr: Union[str, DeviceAddress]) -> DeviceAddress:
        if isinstance(addr, DeviceAddress):
            return addr
        return self.devices.get(addr, DeviceAddress.parse(addr))

    @classmethod
    def create_local_heterogeneous(
        cls,
        num_gpus: int = 1,
        num_cpus: int = 2,
        base_port: int = 9100,
    ) -> "ClusterTopology":
        """
        Creates a topology representing a local cluster with heterogeneous GPU and CPU workers.
        """
        top = cls()
        gpu_devices = [DeviceAddress("node0", "cuda", i) for i in range(num_gpus)]
        cpu_devices = [DeviceAddress("node0", "cpu", i) for i in range(num_cpus)]

        node0 = ClusterNode(
            node_id="node0",
            host="127.0.0.1",
            port=base_port,
            devices=gpu_devices + cpu_devices,
        )
        top.add_node(node0)
        return top

    @classmethod
    def create_multi_node(
        cls,
        node_configs: list[tuple[str, str, int, int, int]],  # (node_id, host, port, num_gpus, num_cpus)
    ) -> "ClusterTopology":
        """
        Creates a multi-node cluster topology.
        """
        top = cls()
        for node_id, host, port, n_gpus, n_cpus in node_configs:
            devs = [DeviceAddress(node_id, "cuda", i) for i in range(n_gpus)]
            devs += [DeviceAddress(node_id, "cpu", i) for i in range(n_cpus)]
            node = ClusterNode(node_id=node_id, host=host, port=port, devices=devs)
            top.add_node(node)
        return top

    def all_devices(self) -> list[DeviceAddress]:
        return list(self.devices.values())

    def gpu_devices(self) -> list[DeviceAddress]:
        return [d for d in self.devices.values() if d.device_type == "cuda"]

    def cpu_devices(self) -> list[DeviceAddress]:
        return [d for d in self.devices.values() if d.device_type == "cpu"]

    def __repr__(self) -> str:
        return f"ClusterTopology(nodes={len(self.nodes)}, devices={len(self.devices)}: {[str(d) for d in self.devices.values()]})"
