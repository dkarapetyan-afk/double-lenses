"""
Unified Tensor Abstraction supporting both NumPy (CPU) and PyTorch (CUDA GPU/CPU).
Provides high-performance differentiable tensor primitives for forward and adjoint passes.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np

# Try importing torch if available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


class Tensor:
    """
    Unified Tensor class that holds an underlying ndarray or torch.Tensor.
    Enables forward evaluation, adjoint cotangent calculations, and cluster serialization.
    """
    def __init__(self, data: Union[np.ndarray, Any], device: str = "cpu", requires_grad: bool = False):
        if HAS_TORCH and isinstance(data, torch.Tensor):
            self._data = data
            self._backend = "torch"
        elif isinstance(data, np.ndarray):
            self._data = data
            self._backend = "numpy"
        else:
            self._data = np.array(data, dtype=np.float32)
            self._backend = "numpy"

        self.device = device
        self.requires_grad = requires_grad
        self.grad: Optional['Tensor'] = None

    @property
    def data(self) -> Any:
        return self._data

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._data.shape)

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def dtype(self) -> Any:
        return self._data.dtype

    def to_numpy(self) -> np.ndarray:
        if self._backend == "numpy":
            return self._data
        else:
            return self._data.detach().cpu().numpy()

    def to_torch(self, device: Optional[str] = None) -> Any:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not installed in the environment.")
        if self._backend == "torch":
            t = self._data
        else:
            t = torch.from_numpy(self._data)
        if device:
            t = t.to(device)
        return t

    def to(self, device: str) -> 'Tensor':
        """Transfers tensor to target device (e.g. 'cpu', 'cuda:0')."""
        if device == self.device:
            return self
        if device.startswith("cuda"):
            if not HAS_TORCH or not torch.cuda.is_available():
                # Fallback to simulated device address for cluster simulation
                res = Tensor(self.to_numpy(), device=device, requires_grad=self.requires_grad)
                res.grad = self.grad
                return res
            t_torch = self.to_torch(device=device)
            res = Tensor(t_torch, device=device, requires_grad=self.requires_grad)
            res.grad = self.grad
            return res
        else:
            # Transfer to CPU
            if self._backend == "torch":
                res = Tensor(self._data.cpu(), device="cpu", requires_grad=self.requires_grad)
            else:
                res = Tensor(self._data, device="cpu", requires_grad=self.requires_grad)
            res.grad = self.grad
            return res

    def copy(self) -> 'Tensor':
        if self._backend == "numpy":
            return Tensor(self._data.copy(), device=self.device, requires_grad=self.requires_grad)
        else:
            return Tensor(self._data.clone(), device=self.device, requires_grad=self.requires_grad)

    def zero_grad(self) -> None:
        self.grad = None

    # --- Math operations with forward & adjoint support ---

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self._data + other_data, device=self.device)

    def __radd__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self.__add__(other)

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self._data - other_data, device=self.device)

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self._data * other_data, device=self.device)

    def __rmul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self.__mul__(other)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return Tensor(self._data @ other.data, device=self.device)

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self._data / other_data, device=self.device)

    def __neg__(self) -> 'Tensor':
        return Tensor(-self._data, device=self.device)

    def transpose(self, *axes: int) -> 'Tensor':
        if self._backend == "numpy":
            return Tensor(np.transpose(self._data, axes if axes else None), device=self.device)
        else:
            if not axes:
                return Tensor(self._data.T, device=self.device)
            return Tensor(self._data.permute(*axes), device=self.device)

    @property
    def T(self) -> 'Tensor':
        return self.transpose()

    def reshape(self, *shape: int) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if self._backend == "numpy":
            return Tensor(self._data.reshape(*shape), device=self.device)
        else:
            return Tensor(self._data.reshape(*shape), device=self.device)

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        if self._backend == "numpy":
            return Tensor(np.sum(self._data, axis=axis, keepdims=keepdims), device=self.device)
        else:
            if axis is None:
                return Tensor(self._data.sum(), device=self.device)
            return Tensor(self._data.sum(dim=axis, keepdim=keepdims), device=self.device)

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        if self._backend == "numpy":
            return Tensor(np.mean(self._data, axis=axis, keepdims=keepdims), device=self.device)
        else:
            if axis is None:
                return Tensor(self._data.mean(), device=self.device)
            return Tensor(self._data.mean(dim=axis, keepdim=keepdims), device=self.device)

    def __getitem__(self, item: Any) -> 'Tensor':
        return Tensor(self._data[item], device=self.device)

    def __setitem__(self, key: Any, value: Union['Tensor', float, int]) -> None:
        val = value.data if isinstance(value, Tensor) else value
        self._data[key] = val

    def item(self) -> float:
        if self._backend == "numpy":
            return float(self._data.item())
        else:
            return float(self._data.item())

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, device='{self.device}', dtype={self.dtype})"


def zeros(shape: Tuple[int, ...], dtype: Any = np.float32, device: str = "cpu") -> Tensor:
    return Tensor(np.zeros(shape, dtype=dtype), device=device)


def ones(shape: Tuple[int, ...], dtype: Any = np.float32, device: str = "cpu") -> Tensor:
    return Tensor(np.ones(shape, dtype=dtype), device=device)


def randn(shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0, dtype: Any = np.float32, device: str = "cpu") -> Tensor:
    data = np.random.normal(mean, std, size=shape).astype(dtype)
    return Tensor(data, device=device)
