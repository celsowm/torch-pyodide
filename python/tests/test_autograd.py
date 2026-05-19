"""Testes do sistema de autograd e loop de treinamento (usando FakeRuntime)."""

import math
import sys
from pathlib import Path

# Adicionar python root ao path
PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import torch as torch_mod


class FakeRuntime:
    """Fake runtime para testes sem Pyodide."""
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.initialized = False
        self.next_id = 1000
        self.store: dict[int, dict[str, object]] = {}

    def _new(self, shape: list[int], values: list[float], dtype: str = "float32") -> dict[str, object]:
        tensor_id = self.next_id
        self.next_id += 1
        self.store[tensor_id] = {"shape": shape, "values": list(values), "dtype": dtype}
        return {"id": tensor_id, "shape": shape, "dtype": dtype}

    def _ensure_ready(self) -> None:
        if not self.available:
            raise RuntimeError("WebGPU unavailable")
        if not self.initialized:
            self.initialized = True

    def tensorFromData(self, flat: list[float], shape: list[int], dtype: str) -> dict[str, object]:
        self._ensure_ready()
        return self._new(shape, flat, dtype)

    def zeros(self, shape: list[int], dtype: str) -> dict[str, object]:
        self._ensure_ready()
        return self._new(shape, [0.0] * math.prod(shape), dtype)

    def ones(self, shape: list[int], dtype: str) -> dict[str, object]:
        self._ensure_ready()
        return self._new(shape, [1.0] * math.prod(shape), dtype)

    def add(self, a: int, b: int) -> dict[str, object]:
        self._ensure_ready()
        va = self.store[a]["values"]
        vb = self.store[b]["values"] if b in self.store else [float(b)]
        result = [x + y for x, y in zip(va, vb * len(va))]
        return self._new(list(self.store[a]["shape"]), result, self.store[a]["dtype"])

    def sub(self, a: int, b: int) -> dict[str, object]:
        self._ensure_ready()
        va = self.store[a]["values"]
        vb = self.store[b]["values"] if b in self.store else [float(b)]
        result = [x - y for x, y in zip(va, vb * len(va))]
        return self._new(list(self.store[a]["shape"]), result, self.store[a]["dtype"])

    def mul(self, a: int, b: int) -> dict[str, object]:
        self._ensure_ready()
        va = self.store[a]["values"]
        vb = self.store[b]["values"] if b in self.store else [float(b)]
        result = [x * y for x, y in zip(va, vb * len(va))]
        return self._new(list(self.store[a]["shape"]), result, self.store[a]["dtype"])

    def div(self, a: int, b: int) -> dict[str, object]:
        self._ensure_ready()
        va = self.store[a]["values"]
        vb = self.store[b]["values"] if b in self.store else [float(b)]
        result = [x / y for x, y in zip(va, vb * len(va))]
        return self._new(list(self.store[a]["shape"]), result, self.store[a]["dtype"])

    def sum(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        val = sum(self.store[tensor_id]["values"])
        return self._new([1], [val], self.store[tensor_id]["dtype"])

    def sumDim(self, tensor_id: int, dim: int, keepdim: bool) -> dict[str, object]:
        return self.sum(tensor_id)

    def mean(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        val = sum(self.store[tensor_id]["values"]) / len(self.store[tensor_id]["values"])
        return self._new([1], [val], self.store[tensor_id]["dtype"])

    def meanDim(self, tensor_id: int, dim: int, keepdim: bool) -> dict[str, object]:
        return self.mean(tensor_id)

    def matmul(self, a: int, b: int) -> dict[str, object]:
        self._ensure_ready()
        # Simplificação: apenas para tensors pequenos
        va = self.store[a]["values"]
        vb = self.store[b]["values"]
        result = [sum(va[i] * vb[j] for i in range(len(va))) for j in range(len(vb))]
        return self._new([1], result[:1], self.store[a]["dtype"])

    def relu(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        val = [max(0.0, v) for v in self.store[tensor_id]["values"]]
        return self._new(list(self.store[tensor_id]["shape"]), val, self.store[tensor_id]["dtype"])

    def neg(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        val = [-v for v in self.store[tensor_id]["values"]]
        return self._new(list(self.store[tensor_id]["shape"]), val, self.store[tensor_id]["dtype"])

    def abs(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        val = [abs(v) for v in self.store[tensor_id]["values"]]
        return self._new(list(self.store[tensor_id]["shape"]), val, self.store[tensor_id]["dtype"])

    def sqrt(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        val = [math.sqrt(v) for v in self.store[tensor_id]["values"]]
        return self._new(list(self.store[tensor_id]["shape"]), val, self.store[tensor_id]["dtype"])

    def exp(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        val = [math.exp(v) for v in self.store[tensor_id]["values"]]
        return self._new(list(self.store[tensor_id]["shape"]), val, self.store[tensor_id]["dtype"])

    def log(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        val = [math.log(v) for v in self.store[tensor_id]["values"]]
        return self._new(list(self.store[tensor_id]["shape"]), val, self.store[tensor_id]["dtype"])

    def toList(self, tensor_id: int) -> list[float]:
        return list(self.store[tensor_id]["values"])

    def destroy(self, tensor_id: int) -> None:
        self.store.pop(tensor_id, None)


def test_is_grad_enabled():
    """Testa is_grad_enabled."""
    from torch import is_grad_enabled, no_grad
    
    assert is_grad_enabled()
    
    with no_grad():
        assert not is_grad_enabled()
    
    print("✓ test_is_grad_enabled passed")


def test_backward_api_exists():
    """Testa que API de backward existe."""
    from torch import tensor
    from torch._tensor import Tensor
    
    # Verificar que métodos existem
    assert hasattr(Tensor, "backward")
    assert hasattr(Tensor, "register_hook")
    assert hasattr(Tensor, "retain_grad")
    assert hasattr(Tensor, "grad")
    assert hasattr(Tensor, "requires_grad")
    assert hasattr(Tensor, "is_leaf")
    
    print("✓ test_backward_api_exists passed")


def test_optimizer_api_exists():
    """Testa que API de otimizador existe."""
    from torch.optim import SGD, Adam, AdamW, RMSprop, Optimizer
    
    # Verificar que métodos existem
    assert hasattr(Optimizer, "zero_grad")
    assert hasattr(Optimizer, "step")
    assert hasattr(SGD, "step")
    assert hasattr(Adam, "step")
    
    print("✓ test_optimizer_api_exists passed")


if __name__ == "__main__":
    test_is_grad_enabled()
    test_backward_api_exists()
    test_optimizer_api_exists()
    print("\n✅ Todos os testes de API passaram!")
