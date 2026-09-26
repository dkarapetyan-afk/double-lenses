"""
Hierarchical Memory Stager and Asynchronous CPU Offloader.
Allows staging large models (Mixtral 8x7B, DeepSeek-V3) across GPU VRAM and CPU RAM.
"""

from double_lenses.autodiff.param_lens import ParameterizedLens
from double_lenses.autodiff.tensor import Tensor
from double_lenses.cluster.topology import DeviceAddress


class MemoryStager:
    """
    Manages asynchronous staging of layer parameters between CPU host memory and GPU VRAM.
    """

    def __init__(self, target_gpu: DeviceAddress, host_cpu: DeviceAddress):
        self.target_gpu = str(target_gpu)
        self.host_cpu = str(host_cpu)
        self.cpu_param_store: dict[str, Tensor] = {}
        self.staged_lenses: list[ParameterizedLens] = []

    def register_for_offload(self, lens: ParameterizedLens) -> None:
        """Stores parameters on CPU RAM and sets up staging hooks."""
        self.staged_lenses.append(lens)
        for k, p in lens.parameters.items():
            self.cpu_param_store[f"{lens.name}.{k}"] = p.to("cpu")

    def prefetch_to_gpu(self, lens: ParameterizedLens) -> None:
        """Prefetches lens parameters from CPU into GPU VRAM."""
        for k, _p in lens.parameters.items():
            full_key = f"{lens.name}.{k}"
            if full_key in self.cpu_param_store:
                cpu_p = self.cpu_param_store[full_key]
                lens.parameters[k] = cpu_p.to(self.target_gpu)

    def evict_to_cpu(self, lens: ParameterizedLens) -> None:
        """Moves parameters and accumulated gradients back to CPU RAM to free GPU VRAM."""
        for k, p in lens.parameters.items():
            full_key = f"{lens.name}.{k}"
            # Synchronize gradient if computed
            if p.grad is not None:
                if self.cpu_param_store[full_key].grad is None:
                    self.cpu_param_store[full_key].grad = p.grad.to("cpu")
                else:
                    self.cpu_param_store[full_key].grad = self.cpu_param_store[full_key].grad + p.grad.to("cpu")
            # Move parameter back to CPU
            lens.parameters[k] = self.cpu_param_store[full_key]
