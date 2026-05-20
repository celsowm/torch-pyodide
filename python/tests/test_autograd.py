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
        batch, in_ch, in_h, in_w = inp["shape"]
        out_ch, _, kh, kw = w["shape"]
        s_h, s_w = stride[0], stride[1] if len(stride) > 1 else stride[0]
        p_h, p_w = padding[0], padding[1] if len(padding) > 1 else padding[0]
        out_h = (in_h + 2 * p_h - kh) // s_h + 1
        out_w = (in_w + 2 * p_w - kw) // s_w + 1
        total = batch * out_ch * out_h * out_w
        return self._new([batch, out_ch, out_h, out_w], [0.5] * total, "float32")

    def sigmoid(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        result = [1.0 / (1.0 + math.exp(-v)) for v in vals]
        return self._new(list(self.store[tensor_id]["shape"]), result, self.store[tensor_id]["dtype"])

    def tanh(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        result = [math.tanh(v) for v in vals]
        return self._new(list(self.store[tensor_id]["shape"]), result, self.store[tensor_id]["dtype"])

    def gelu(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        result = [0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in vals]
        return self._new(list(self.store[tensor_id]["shape"]), result, self.store[tensor_id]["dtype"])

    def silu(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        result = [v / (1.0 + math.exp(-v)) for v in vals]
        return self._new(list(self.store[tensor_id]["shape"]), result, self.store[tensor_id]["dtype"])

    def leakyRelu(self, tensor_id: int, alpha: float = 0.01) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        result = [v if v >= 0 else alpha * v for v in vals]
        return self._new(list(self.store[tensor_id]["shape"]), result, self.store[tensor_id]["dtype"])

    def expand(self, tensor_id: int, shape: list[int]) -> dict[str, object]:
        self._ensure_ready()
        original = self.store[tensor_id]["values"]
        total = max(1, math.prod(shape))
        result = []
        for i in range(total):
            result.append(original[i % len(original)])
        return self._new(shape, result, self.store[tensor_id]["dtype"])

    def where(self, cond_id: int, x_id: int, y_id: int) -> dict[str, object]:
        self._ensure_ready()
        cond = self.store[cond_id]["values"]
        xv = self.store[x_id]["values"]
        yv = self.store[y_id]["values"]
        result = [xv[i] if bool(cond[i]) else yv[i] for i in range(len(xv))]
        return self._new(list(self.store[x_id]["shape"]), result, self.store[x_id]["dtype"])

    def cat(self, ids: list[int], dim: int) -> dict[str, object]:
        self._ensure_ready()
        all_vals = []
        for tid in ids:
            all_vals.extend(self.store[tid]["values"])
        shape = list(self.store[ids[0]]["shape"])
        total_vals = sum(len(self.store[tid]["values"]) for tid in ids)
        new_shape = shape.copy()
        if dim < 0:
            dim += len(shape)
        if dim < len(shape):
            new_shape[dim] = total_vals // max(1, math.prod(shape[:dim] + shape[dim+1:]))
        return self._new(new_shape, all_vals, self.store[ids[0]]["dtype"])

    def cumsum(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        result = []
        s = 0.0
        for v in vals:
            s += v
            result.append(s)
        return self._new(list(self.store[tensor_id]["shape"]), result, self.store[tensor_id]["dtype"])

    def cumprod(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        result = []
        p = 1.0
        for v in vals:
            p *= v
            result.append(p)
        return self._new(list(self.store[tensor_id]["shape"]), result, self.store[tensor_id]["dtype"])

    def tril(self, tensor_id: int, diagonal: int) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        shape = self.store[tensor_id]["shape"]
        result = list(vals)
        if len(shape) == 2:
            rows, cols = shape
            for r in range(rows):
                for c in range(cols):
                    if c > r + diagonal:
                        result[r * cols + c] = 0.0
        return self._new(shape, result, self.store[tensor_id]["dtype"])

    def triu(self, tensor_id: int, diagonal: int) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        shape = self.store[tensor_id]["shape"]
        result = list(vals)
        if len(shape) == 2:
            rows, cols = shape
            for r in range(rows):
                for c in range(cols):
                    if c < r + diagonal:
                        result[r * cols + c] = 0.0
        return self._new(shape, result, self.store[tensor_id]["dtype"])

    def flip(self, tensor_id: int, dims: list[int]) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        return self._new(list(self.store[tensor_id]["shape"]), list(reversed(vals)), self.store[tensor_id]["dtype"])

    def maskedSelect(self, tensor_id: int, mask_id: int) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        mask = self.store[mask_id]["values"]
        result = [v for v, m in zip(vals, mask) if bool(m)]
        return self._new([len(result)], result, self.store[tensor_id]["dtype"])

    def maskedFill(self, tensor_id: int, mask_id: int, value: float) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        mask = self.store[mask_id]["values"]
        result = [value if bool(m) else v for v, m in zip(vals, mask)]
        return self._new(list(self.store[tensor_id]["shape"]), result, self.store[tensor_id]["dtype"])

    def maximum(self, a: int, b: int) -> dict[str, object]:
        self._ensure_ready()
        va = self.store[a]["values"]
        vb = self.store[b]["values"]
        result = [max(x, y) for x, y in zip(va, vb)]
        return self._new(list(self.store[a]["shape"]), result, self.store[a]["dtype"])

    def minimum(self, a: int, b: int) -> dict[str, object]:
        self._ensure_ready()
        va = self.store[a]["values"]
        vb = self.store[b]["values"]
        result = [min(x, y) for x, y in zip(va, vb)]
        return self._new(list(self.store[a]["shape"]), result, self.store[a]["dtype"])

    def sort(self, tensor_id: int, dim: int) -> list[dict[str, object]]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        indexed = sorted(enumerate(vals), key=lambda x: x[1])
        sorted_vals = [v for _, v in indexed]
        sorted_idxs = [float(i) for i, _ in indexed]
        shape = list(self.store[tensor_id]["shape"])
        dtype = self.store[tensor_id]["dtype"]
        m1 = self._new(shape, sorted_vals, dtype)
        m2 = self._new(shape, sorted_idxs, "int64")
        return [m1, m2]

    def slice(self, tensor_id: int, dim: int, start: int, end: int, step: int) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        shape = list(self.store[tensor_id]["shape"])
        sliced = vals[start:end:step]
        new_shape = shape.copy()
        if dim < len(new_shape):
            new_shape[dim] = len(sliced) // max(1, math.prod(new_shape[:dim] + new_shape[dim+1:]))
        return self._new(new_shape, sliced, self.store[tensor_id]["dtype"])

    def sign(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        result = [1.0 if v > 0 else (-1.0 if v < 0 else 0.0) for v in vals]
        return self._new(list(self.store[tensor_id]["shape"]), result, self.store[tensor_id]["dtype"])

    def gt(self, a: int, b: int) -> dict[str, object]:
        self._ensure_ready()
        va = self.store[a]["values"]
        vb = self.store[b]["values"] if b in self.store else [float(b)]
        result = [1.0 if x > y else 0.0 for x, y in zip(va, vb * len(va))]
        return self._new(list(self.store[a]["shape"]), result, "float32")

    def lt(self, a: int, b: int) -> dict[str, object]:
        self._ensure_ready()
        va = self.store[a]["values"]
        vb = self.store[b]["values"] if b in self.store else [float(b)]
        result = [1.0 if x < y else 0.0 for x, y in zip(va, vb * len(va))]
        return self._new(list(self.store[a]["shape"]), result, "float32")

    def ge(self, a: int, b: int) -> dict[str, object]:
        self._ensure_ready()
        va = self.store[a]["values"]
        vb = self.store[b]["values"] if b in self.store else [float(b)]
        result = [1.0 if x >= y else 0.0 for x, y in zip(va, vb * len(va))]
        return self._new(list(self.store[a]["shape"]), result, "float32")

    def le(self, a: int, b: int) -> dict[str, object]:
        self._ensure_ready()
        va = self.store[a]["values"]
        vb = self.store[b]["values"] if b in self.store else [float(b)]
        result = [1.0 if x <= y else 0.0 for x, y in zip(va, vb * len(va))]
        return self._new(list(self.store[a]["shape"]), result, "float32")

    def clone(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        return self._new(list(self.store[tensor_id]["shape"]), list(self.store[tensor_id]["values"]), self.store[tensor_id]["dtype"])

    def to(self, tensor_id: int, dtype: str) -> dict[str, object]:
        self._ensure_ready()
        return self._new(list(self.store[tensor_id]["shape"]), list(self.store[tensor_id]["values"]), dtype)

    def pow(self, a: int, b: int) -> dict[str, object]:
        self._ensure_ready()
        va = self.store[a]["values"]
        vb = self.store[b]["values"] if b in self.store else [float(b)]
        result = [x ** y for x, y in zip(va, vb * len(va))]
        return self._new(list(self.store[a]["shape"]), result, self.store[a]["dtype"])

    def prod(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        p = 1.0
        for v in vals:
            p *= v
        return self._new([1], [p], self.store[tensor_id]["dtype"])

    def reshape(self, tensor_id: int, shape: list[int]) -> dict[str, object]:
        self._ensure_ready()
        return self._new(shape, list(self.store[tensor_id]["values"]), self.store[tensor_id]["dtype"])

    def transpose(self, tensor_id: int, dim0: int, dim1: int) -> dict[str, object]:
        self._ensure_ready()
        return self._new(list(reversed(self.store[tensor_id]["shape"])), list(self.store[tensor_id]["values"]), self.store[tensor_id]["dtype"])

    def squeeze(self, tensor_id: int) -> dict[str, object]:
        self._ensure_ready()
        shape = [s for s in self.store[tensor_id]["shape"] if s != 1]
        if not shape:
            shape = [1]
        return self._new(shape, list(self.store[tensor_id]["values"]), self.store[tensor_id]["dtype"])

    def select(self, tensor_id: int, dim: int, index: int) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        return self._new([1], [vals[index] if index < len(vals) else 0.0], self.store[tensor_id]["dtype"])

    def sliceBackward(self, grad_output_id: int, input_shape: list[int],
                      sliced_shape: list[int], dim: int, start: int, step: int) -> dict[str, object]:
        self._ensure_ready()
        total = math.prod(input_shape)
        return self._new(list(input_shape), [0.0] * total, "float32")

    def fill(self, tensor_id: int, value: float) -> dict[str, object]:
        self._ensure_ready()
        vals = self.store[tensor_id]["values"]
        return self._new(list(self.store[tensor_id]["shape"]), [value] * len(vals), self.store[tensor_id]["dtype"])


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

    print("[OK] test_conv2d_backward_functions_exist passed")


def test_nll_loss_backward_functions_exist():
    """Testa que funções de backward para NLL loss existem."""
    from torch._tensor import nll_loss_backward_from_tensors

    assert callable(nll_loss_backward_from_tensors)

    print("[OK] test_nll_loss_backward_functions_exist passed")


def test_log_softmax_backward_functions_exist():
    """Testa que funções de backward para log_softmax existem."""
    from torch._tensor import log_softmax_backward_from_tensors

    assert callable(log_softmax_backward_from_tensors)

    print("[OK] test_log_softmax_backward_functions_exist passed")


@pytest.fixture
def fake_runtime():
    """Fixture para injetar FakeRuntime nos testes."""
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    return fake


def test_conv2d_creates_computational_node():
    """Testa que conv2d cria nó computacional quando requires_grad=True."""
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value

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

    print("[OK] test_conv2d_creates_computational_node passed")


def test_nll_loss_creates_computational_node():
    """Testa que nll_loss cria nó computacional quando requires_grad=True."""
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value

    from torch._tensor import nll_loss_from_tensor
    from torch import tensor

    logits = tensor([0.5, -0.5, 0.1, -0.3, 0.8, 0.2], requires_grad=True)
    targets = tensor([0, 1], dtype="int32")

    loss = nll_loss_from_tensor(logits, targets)

    assert loss._requires_grad

    print("[OK] test_nll_loss_creates_computational_node passed")


def test_log_softmax_creates_computational_node():
    """Testa que log_softmax cria nó computacional quando requires_grad=True."""
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value

    from torch._tensor import log_softmax_from_tensor
    from torch import tensor

    x = tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)

    out = log_softmax_from_tensor(x, dim=-1)

    assert out._requires_grad

    print("[OK] test_log_softmax_creates_computational_node passed")


def test_sigmoid_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor, sigmoid
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = sigmoid(x)
    assert out._requires_grad
    print("[OK] test_sigmoid_creates_computational_node passed")


def test_tanh_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor, tanh
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = tanh(x)
    assert out._requires_grad
    print("[OK] test_tanh_creates_computational_node passed")


def test_gelu_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = x.gelu()
    assert out._requires_grad
    print("[OK] test_gelu_creates_computational_node passed")


def test_silu_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = x.silu()
    assert out._requires_grad
    print("[OK] test_silu_creates_computational_node passed")


def test_leaky_relu_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = x.leaky_relu(0.1)
    assert out._requires_grad
    print("[OK] test_leaky_relu_creates_computational_node passed")


def test_softmax_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor
    x = tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    out = x.softmax(dim=-1)
    assert out._requires_grad
    print("[OK] test_softmax_creates_computational_node passed")


def test_expand_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor, expand
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = expand(x, [3, 3])
    assert out._requires_grad
    print("[OK] test_expand_creates_computational_node passed")


def test_where_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor, where
    cond = tensor([1.0, 0.0, 1.0])
    x = tensor([10.0, 20.0, 30.0], requires_grad=True)
    y = tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = where(cond, x, y)
    assert out._requires_grad
    print("[OK] test_where_creates_computational_node passed")


def test_cat_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor, cat
    a = tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = tensor([4.0, 5.0, 6.0], requires_grad=True)
    out = cat([a, b])
    assert out._requires_grad
    print("[OK] test_cat_creates_computational_node passed")


def test_cumsum_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = x.cumsum()
    assert out._requires_grad
    print("[OK] test_cumsum_creates_computational_node passed")


def test_cumprod_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = x.cumprod()
    assert out._requires_grad
    print("[OK] test_cumprod_creates_computational_node passed")


def test_tril_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor
    x = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    out = x.tril()
    assert out._requires_grad
    print("[OK] test_tril_creates_computational_node passed")


def test_triu_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor
    x = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    out = x.triu()
    assert out._requires_grad
    print("[OK] test_triu_creates_computational_node passed")


def test_flip_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor
    x = tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    out = x.flip([0])
    assert out._requires_grad
    print("[OK] test_flip_creates_computational_node passed")


def test_masked_select_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor
    x = tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    mask = tensor([1.0, 0.0, 1.0, 0.0])
    out = x.masked_select(mask)
    assert out._requires_grad
    print("[OK] test_masked_select_creates_computational_node passed")


def test_masked_fill_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor
    x = tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    mask = tensor([1.0, 0.0, 1.0, 0.0])
    out = x.masked_fill(mask, 0.0)
    assert out._requires_grad
    print("[OK] test_masked_fill_creates_computational_node passed")


def test_maximum_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor, maximum
    a = tensor([1.0, 5.0, 3.0], requires_grad=True)
    b = tensor([4.0, 2.0, 6.0], requires_grad=True)
    out = maximum(a, b)
    assert out._requires_grad
    print("[OK] test_maximum_creates_computational_node passed")


def test_minimum_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor, minimum
    a = tensor([1.0, 5.0, 3.0], requires_grad=True)
    b = tensor([4.0, 2.0, 6.0], requires_grad=True)
    out = minimum(a, b)
    assert out._requires_grad
    print("[OK] test_minimum_creates_computational_node passed")


def test_sort_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor
    x = tensor([3.0, 1.0, 2.0], requires_grad=True)
    values, indices = x.sort()
    assert values._requires_grad
    print("[OK] test_sort_creates_computational_node passed")


def test_topk_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor
    x = tensor([3.0, 1.0, 2.0, 5.0, 4.0], requires_grad=True)
    values, indices = x.topk(3)
    assert values._requires_grad
    print("[OK] test_topk_creates_computational_node passed")


def test_scatter_creates_computational_node():
    fake = FakeRuntime()
    tensor_mod._get_runtime = lambda: fake
    tensor_mod._run_js_awaitable = lambda value: value
    from torch import tensor
    x = tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    idx = tensor([1, 3], dtype="int32")
    out = x.scatter_(0, idx, 0.0)
    assert out._requires_grad
    print("[OK] test_scatter_creates_computational_node passed")


if __name__ == "__main__":
    test_conv2d_backward_functions_exist()
    test_nll_loss_backward_functions_exist()
    test_log_softmax_backward_functions_exist()
    test_conv2d_creates_computational_node()
    test_nll_loss_creates_computational_node()
    test_log_softmax_creates_computational_node()
    test_sigmoid_creates_computational_node()
    test_tanh_creates_computational_node()
    test_gelu_creates_computational_node()
    test_silu_creates_computational_node()
    test_leaky_relu_creates_computational_node()
    test_softmax_creates_computational_node()
    test_expand_creates_computational_node()
    test_where_creates_computational_node()
    test_cat_creates_computational_node()
    test_cumsum_creates_computational_node()
    test_cumprod_creates_computational_node()
    test_tril_creates_computational_node()
    test_triu_creates_computational_node()
    test_flip_creates_computational_node()
    test_masked_select_creates_computational_node()
    test_masked_fill_creates_computational_node()
    test_maximum_creates_computational_node()
    test_minimum_creates_computational_node()
    test_sort_creates_computational_node()
    test_topk_creates_computational_node()
    test_scatter_creates_computational_node()
    print("\n[OK] Todos os testes de API passaram!")
