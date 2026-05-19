"""Testes do sistema de autograd e loop de treinamento (usando FakeRuntime)."""

import math
import sys
from pathlib import Path

import pytest

# Adicionar python root ao path
PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import torch as torch_mod
import torch._tensor as tensor_mod


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

    # ─── Backward methods for runtime-backed tests ───────────────────────────

    def conv2dInputBackward(self, grad_output_id: int, weight_id: int,
                            input_shape: list[int], grad_output_shape: list[int],
                            stride: list[int], padding: list[int]) -> dict[str, object]:
        self._ensure_ready()
        total = math.prod(input_shape)
        return self._new(list(input_shape), [0.1] * total, "float32")

    def conv2dWeightBackward(self, grad_output_id: int, input_id: int,
                             weight_shape: list[int], grad_output_shape: list[int],
                             input_shape: list[int], stride: list[int], padding: list[int]) -> dict[str, object]:
        self._ensure_ready()
        total = math.prod(weight_shape)
        return self._new(list(weight_shape), [0.1] * total, "float32")

    def conv2dBiasBackward(self, grad_output_id: int, out_ch: int,
                           grad_output_shape: list[int]) -> dict[str, object]:
        self._ensure_ready()
        return self._new([out_ch], [0.1] * out_ch, "float32")

    def logSoftmaxBackward(self, grad_output_id: int, softmax_id: int,
                           batch_size: int, num_classes: int) -> dict[str, object]:
        self._ensure_ready()
        total = batch_size * num_classes
        return self._new([batch_size, num_classes], [0.05] * total, "float32")

    def nllLossBackward(self, targets_id: int, batch_size: int,
                        num_classes: int, scale: float = 1.0) -> dict[str, object]:
        self._ensure_ready()
        total = batch_size * num_classes
        return self._new([batch_size, num_classes], [-scale] * total, "float32")

    def softmax(self, tensor_id: int, dim: int) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        max_v = max(vals)
        exps = [math.exp(v - max_v) for v in vals]
        s = sum(exps)
        softmax_vals = [e / s for e in exps]
        return self._new(list(self.store[tensor_id]["shape"]), softmax_vals, self.store[tensor_id]["dtype"])

    def logSoftmax(self, tensor_id: int, dim: int) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        max_v = max(vals)
        exps = [math.exp(v - max_v) for v in vals]
        s = sum(exps)
        log_softmax_vals = [v - max_v - math.log(s) for v in vals]
        return self._new(list(self.store[tensor_id]["shape"]), log_softmax_vals, self.store[tensor_id]["dtype"])

    def nllLoss(self, input_id: int, targets_id: int) -> dict[str, object]:
        self._ensure_ready()
        inp = self.store[input_id]
        targets = self.store[targets_id]["values"]
        batch_size = len(targets)
        num_classes = inp["shape"][-1] if len(inp["shape"]) > 1 else 1
        # Simplified: just return mean of negative log probs
        vals = inp["values"]
        loss_vals = []
        for b in range(batch_size):
            target_idx = int(targets[b])
            idx = b * num_classes + target_idx
            if idx < len(vals):
                loss_vals.append(-vals[idx])
        avg_loss = sum(loss_vals) / len(loss_vals) if loss_vals else 0.0
        return self._new([1], [avg_loss], "float32")

    def conv2d(self, input_id: int, weight_id: int, bias: list[float] | None,
               stride: list[int], padding: list[int], dilation: list[int],
               groups: int) -> dict[str, object]:
        self._ensure_ready()
        inp = self.store[input_id]
        w = self.store[weight_id]
        # Simplified: output shape for conv2d
        batch, in_ch, in_h, in_w = inp["shape"]
        out_ch, _, kh, kw = w["shape"]
        s_h, s_w = stride[0], stride[1] if len(stride) > 1 else stride[0]
        p_h, p_w = padding[0], padding[1] if len(padding) > 1 else padding[0]
        out_h = (in_h + 2 * p_h - kh) // s_h + 1
        out_w = (in_w + 2 * p_w - kw) // s_w + 1
        total = batch * out_ch * out_h * out_w
        return self._new([batch, out_ch, out_h, out_w], [0.5] * total, "float32")


def test_conv2d_backward_functions_exist():
    """Testa que funções de backward para conv2d existem e são chamáveis."""
    from torch._tensor import (
        conv2d_input_backward_from_tensors,
        conv2d_weight_backward_from_tensors,
        conv2d_bias_backward_from_tensors,
    )
    from torch._tensor import Tensor

    assert callable(conv2d_input_backward_from_tensors)
    assert callable(conv2d_weight_backward_from_tensors)
    assert callable(conv2d_bias_backward_from_tensors)

    print("✓ test_conv2d_backward_functions_exist passed")


def test_nll_loss_backward_functions_exist():
    """Testa que funções de backward para NLL loss existem."""
    from torch._tensor import nll_loss_backward_from_tensors

    assert callable(nll_loss_backward_from_tensors)

    print("✓ test_nll_loss_backward_functions_exist passed")


def test_log_softmax_backward_functions_exist():
    """Testa que funções de backward para log_softmax existem."""
    from torch._tensor import log_softmax_backward_from_tensors

    assert callable(log_softmax_backward_from_tensors)

    print("✓ test_log_softmax_backward_functions_exist passed")


@pytest.fixture
def fake_runtime(monkeypatch):
    """Fixture para injetar FakeRuntime nos testes."""
    fake = FakeRuntime()
    monkeypatch.setattr(tensor_mod, "_get_runtime", lambda: fake)
    monkeypatch.setattr(tensor_mod, "_run_js_awaitable", lambda value: value)
    return fake


def test_conv2d_creates_computational_node(monkeypatch):
    """Testa que conv2d cria nó computacional quando requires_grad=True."""
    fake = FakeRuntime()
    monkeypatch.setattr(tensor_mod, "_get_runtime", lambda: fake)
    monkeypatch.setattr(tensor_mod, "_run_js_awaitable", lambda value: value)

    from torch._tensor import conv2d_from_tensors, Tensor, tensor_from_data
    from torch import tensor

    # Create tensors with correct 4D shape for conv2d: (N, C, H, W)
    x_data = [float(i) for i in range(1 * 2 * 2 * 2)]  # 8 values
    x = tensor_from_data(x_data, shape=[1, 2, 2, 2], requires_grad=True)
    w_data = [1.0, 0.0, 0.0, 1.0]  # 4 values
    w = tensor_from_data(w_data, shape=[1, 2, 1, 2], requires_grad=True)

    out = conv2d_from_tensors(x, w, stride=(1, 1), padding=(0, 0))

    # Output should have requires_grad
    assert out._requires_grad

    print("✓ test_conv2d_creates_computational_node passed")


def test_nll_loss_creates_computational_node(monkeypatch):
    """Testa que nll_loss cria nó computacional quando requires_grad=True."""
    fake = FakeRuntime()
    monkeypatch.setattr(tensor_mod, "_get_runtime", lambda: fake)
    monkeypatch.setattr(tensor_mod, "_run_js_awaitable", lambda value: value)

    from torch._tensor import nll_loss_from_tensor
    from torch import tensor

    logits = tensor([0.5, -0.5, 0.1, -0.3, 0.8, 0.2], requires_grad=True)
    targets = tensor([0, 1], dtype="int32")

    loss = nll_loss_from_tensor(logits, targets)

    assert loss._requires_grad

    print("✓ test_nll_loss_creates_computational_node passed")


def test_log_softmax_creates_computational_node(monkeypatch):
    """Testa que log_softmax cria nó computacional quando requires_grad=True."""
    fake = FakeRuntime()
    monkeypatch.setattr(tensor_mod, "_get_runtime", lambda: fake)
    monkeypatch.setattr(tensor_mod, "_run_js_awaitable", lambda value: value)

    from torch._tensor import log_softmax_from_tensor
    from torch import tensor

    x = tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)

    out = log_softmax_from_tensor(x, dim=-1)

    assert out._requires_grad

    print("✓ test_log_softmax_creates_computational_node passed")


if __name__ == "__main__":
    test_is_grad_enabled()
    test_backward_api_exists()
    test_optimizer_api_exists()
    test_conv2d_backward_functions_exist()
    test_nll_loss_backward_functions_exist()
    test_log_softmax_backward_functions_exist()
    test_conv2d_creates_computational_node()
    test_nll_loss_creates_computational_node()
    test_log_softmax_creates_computational_node()
    print("\n✅ Todos os testes de API passaram!")
