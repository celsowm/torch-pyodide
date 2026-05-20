"""Testes para os gradientes _grad_select, _grad_max, _grad_min, _grad_masked_select, _grad_index_select."""
import math
import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import torch as torch_mod
import torch._tensor as tensor_mod
from torch._tensor import Tensor, tensor_from_data, zeros_like_from_tensor


class FakeRuntime:
    def __init__(self) -> None:
        self.available = True
        self.initialized = False
        self.next_id = 1000
        self.store: dict[int, dict] = {}

    def _new(self, shape, values, dtype="float32"):
        tid = self.next_id; self.next_id += 1
        self.store[tid] = {"shape": shape, "values": list(values), "dtype": dtype}
        return {"id": tid, "shape": shape, "dtype": dtype}

    def _ensure_ready(self):
        self.initialized = True

    def tensorFromData(self, flat, shape, dtype):
        self._ensure_ready()
        return self._new(shape, flat, dtype)

    def zeros(self, shape, dtype):
        self._ensure_ready()
        return self._new(shape, [0.0] * max(1, math.prod(shape) if shape else 1), dtype)

    def ones(self, shape, dtype):
        self._ensure_ready()
        return self._new(shape, [1.0] * max(1, math.prod(shape) if shape else 1), dtype)

    def zerosLike(self, tid, dtype=None):
        t = self.store[tid]
        sz = max(1, math.prod(list(t["shape"])) if t["shape"] else 1)
        dt = str(t["dtype"]) if dtype is None else dtype
        return self._new(list(t["shape"]), [0.0] * sz, dt)

    def onesLike(self, tid, dtype=None):
        t = self.store[tid]
        sz = max(1, math.prod(list(t["shape"])) if t["shape"] else 1)
        dt = str(t["dtype"]) if dtype is None else dtype
        return self._new(list(t["shape"]), [1.0] * sz, dt)

    def add(self, a, b):
        va = self.store[a]["values"]
        vb = [float(b)] if isinstance(b, int) else self.store[b]["values"]
        if len(va) != len(vb):
            if len(vb) == 1:
                vb = vb * len(va)
            elif len(va) == 1:
                va = va * len(vb)
        return self._new(list(self.store[a]["shape"]), [x + y for x, y in zip(va, vb)], self.store[a]["dtype"])

    def sub(self, a, b):
        va = self.store[a]["values"]
        vb = [float(b)] if isinstance(b, int) else self.store[b]["values"]
        return self._new(list(self.store[a]["shape"]), [x - y for x, y in zip(va, vb)], self.store[a]["dtype"])

    def mul(self, a, b):
        va = self.store[a]["values"]
        vb = [float(b)] if isinstance(b, int) else self.store[b]["values"]
        return self._new(list(self.store[a]["shape"]), [x * y for x, y in zip(va, vb)], self.store[a]["dtype"])

    def div(self, a, b):
        va = self.store[a]["values"]
        vb = [float(b)] if isinstance(b, int) else self.store[b]["values"]
        return self._new(list(self.store[a]["shape"]), [x / y for x, y in zip(va, vb)], self.store[a]["dtype"])

    def sum(self, tid):
        return self._new([], [sum(self.store[tid]["values"])], self.store[tid]["dtype"])

    def mean(self, tid):
        v = self.store[tid]["values"]
        return self._new([], [sum(v) / len(v)], self.store[tid]["dtype"])

    def matmul(self, a, b):
        return self._new([1], [0.0], self.store[a]["dtype"])

    def relu(self, tid):
        return self._new(list(self.store[tid]["shape"]), [max(0, v) for v in self.store[tid]["values"]], self.store[tid]["dtype"])

    def neg(self, tid):
        return self._new(list(self.store[tid]["shape"]), [-v for v in self.store[tid]["values"]], self.store[tid]["dtype"])

    def abs(self, tid):
        return self._new(list(self.store[tid]["shape"]), [abs(v) for v in self.store[tid]["values"]], self.store[tid]["dtype"])

    def sqrt(self, tid):
        return self._new(list(self.store[tid]["shape"]), [math.sqrt(v) for v in self.store[tid]["values"]], self.store[tid]["dtype"])

    def exp(self, tid):
        return self._new(list(self.store[tid]["shape"]), [math.exp(v) for v in self.store[tid]["values"]], self.store[tid]["dtype"])

    def log(self, tid):
        return self._new(list(self.store[tid]["shape"]), [math.log(v) for v in self.store[tid]["values"]], self.store[tid]["dtype"])

    def toList(self, tid):
        return list(self.store[tid]["values"])

    def destroy(self, tid):
        self.store.pop(tid, None)

    def argmax(self, tid):
        vals = self.store[tid]["values"]
        idx = max(range(len(vals)), key=lambda i: vals[i])
        return self._new([], [float(idx)], "int32")

    def argmin(self, tid):
        vals = self.store[tid]["values"]
        idx = min(range(len(vals)), key=lambda i: vals[i])
        return self._new([], [float(idx)], "int32")

    def select(self, tid, dim, index):
        t = self.store[tid]
        shape = list(t["shape"])
        d = dim if dim >= 0 else dim + len(shape)
        out_shape = shape[:d] + shape[d+1:]
        stride = 1
        for s in reversed(shape[d+1:]):
            stride *= s
        outer = 1
        for s in shape[:d]:
            outer *= s
        vals = t["values"]
        out = []
        for o in range(outer):
            for s in range(stride):
                idx = o * shape[d] * stride + index * stride + s
                out.append(vals[idx])
        return self._new(out_shape, out, str(t["dtype"]))

    def slice(self, tid, dim, start=None, end=None, step=1):
        t = self.store[tid]
        shape = list(t["shape"])
        d = dim if dim >= 0 else dim + len(shape)
        rng = list(range(shape[d]))[slice(start, end, step)]
        out_shape = list(shape)
        out_shape[d] = len(rng)
        stride = 1
        for s in reversed(shape[d+1:]):
            stride *= s
        outer = 1
        for s in shape[:d]:
            outer *= s
        vals = t["values"]
        out = []
        for o in range(outer):
            for i in rng:
                for s in range(stride):
                    idx = o * shape[d] * stride + i * stride + s
                    out.append(vals[idx])
        return self._new(out_shape, out, str(t["dtype"]))

    def reshape(self, tid, shape):
        t = self.store[tid]
        return self._new(shape, list(t["values"]), str(t["dtype"]))

    def flatten(self, tid, start_dim=0, end_dim=-1):
        t = self.store[tid]
        shape = list(t["shape"])
        return self._new([math.prod(shape)], list(t["values"]), str(t["dtype"]))

    def squeeze(self, tid, dim=None):
        t = self.store[tid]
        shape = list(t["shape"])
        if dim is None:
            out_shape = [d for d in shape if d != 1]
        else:
            out_shape = list(shape)
            if out_shape[dim] == 1:
                out_shape.pop(dim)
        return self._new(out_shape, list(t["values"]), str(t["dtype"]))

    def unsqueeze(self, tid, dim):
        t = self.store[tid]
        shape = list(t["shape"])
        d = dim if dim >= 0 else dim + len(shape) + 1
        out_shape = list(shape[:d]) + [1] + list(shape[d:])
        return self._new(out_shape, list(t["values"]), str(t["dtype"]))

    def transpose(self, tid, dim0, dim1):
        return self.permute(tid, list(range(len(self.store[tid]["shape"]))))

    def permute(self, tid, dims):
        t = self.store[tid]
        return self._new(list(t["shape"]), list(t["values"]), str(t["dtype"]))

    def sigmoid(self, tid):
        t = self.store[tid]
        return self._new(list(t["shape"]), [1/(1+math.exp(-v)) for v in t["values"]], str(t["dtype"]))

    def tanh(self, tid):
        t = self.store[tid]
        return self._new(list(t["shape"]), [math.tanh(v) for v in t["values"]], str(t["dtype"]))

    def gelu(self, tid):
        t = self.store[tid]
        return self._new(list(t["shape"]), list(t["values"]), str(t["dtype"]))

    def silu(self, tid):
        t = self.store[tid]
        return self._new(list(t["shape"]), [v/(1+math.exp(-v)) for v in t["values"]], str(t["dtype"]))

    def leakyRelu(self, tid, alpha=0.01):
        t = self.store[tid]
        return self._new(list(t["shape"]), [max(alpha*v, v) for v in t["values"]], str(t["dtype"]))

    def softmax(self, tid, dim):
        t = self.store[tid]
        return self._new(list(t["shape"]), list(t["values"]), str(t["dtype"]))

    def logSoftmax(self, tid, dim):
        t = self.store[tid]
        return self._new(list(t["shape"]), list(t["values"]), str(t["dtype"]))

    def nllLoss(self, input_id, targets_id):
        inp = self.store[input_id]
        targets = self.store[targets_id]["values"]
        batch_size = len(targets)
        num_classes = inp["shape"][-1] if len(inp["shape"]) > 1 else 1
        vals = inp["values"]
        loss_vals = []
        for b in range(batch_size):
            target_idx = int(targets[b])
            idx = b * num_classes + target_idx
            if idx < len(vals):
                loss_vals.append(-vals[idx])
        avg_loss = sum(loss_vals) / len(loss_vals) if loss_vals else 0.0
        return self._new([1], [avg_loss], "float32")

    def conv2d(self, input_id, weight_id, bias, stride, padding, dilation, groups):
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

    def conv2dInputBackward(self, grad_output_id, weight_id, input_shape, grad_output_shape, stride, padding):
        return self._new(list(input_shape), [0.1] * math.prod(input_shape), "float32")

    def conv2dWeightBackward(self, grad_output_id, input_id, weight_shape, grad_output_shape, input_shape, stride, padding):
        return self._new(list(weight_shape), [0.1] * math.prod(weight_shape), "float32")

    def conv2dBiasBackward(self, grad_output_id, out_ch, grad_output_shape):
        return self._new([out_ch], [0.1] * out_ch, "float32")

    def nllLossBackward(self, targets_id, batch_size, num_classes, scale=1.0):
        return self._new([batch_size, num_classes], [-scale] * (batch_size * num_classes), "float32")

    def logSoftmaxBackward(self, grad_output_id, softmax_id, batch_size, num_classes):
        return self._new([batch_size, num_classes], [0.05] * (batch_size * num_classes), "float32")

    def sumDim(self, tid, dim, keepdim):
        return self._new([1], [sum(self.store[tid]["values"])], self.store[tid]["dtype"])

    def meanDim(self, tid, dim, keepdim):
        v = self.store[tid]["values"]
        return self._new([1], [sum(v)/len(v)], self.store[tid]["dtype"])

    def clamp(self, tid, min_val, max_val):
        t = self.store[tid]
        return self._new(list(t["shape"]), [max(min(v, max_val), min_val) for v in t["values"]], str(t["dtype"]))

    def where(self, cond_id, x_id, y_id):
        cond = self.store[cond_id]["values"]
        x = self.store[x_id]["values"]
        y = self.store[y_id]["values"]
        out = [xv if float(c) > 0 else yv for c, xv, yv in zip(cond, x, y)]
        return self._new(list(self.store[x_id]["shape"]), out, str(self.store[x_id]["dtype"]))

    def max(self, tid):
        t = self.store[tid]
        vals = t["values"]
        mx = max(vals) if vals else 0.0
        return self._new([], [mx], str(t["dtype"]))

    def min(self, tid):
        t = self.store[tid]
        vals = t["values"]
        mn = min(vals) if vals else 0.0
        return self._new([], [mn], str(t["dtype"]))

    def floor(self, tid):
        return self._new(list(self.store[tid]["shape"]), [math.floor(v) for v in self.store[tid]["values"]], self.store[tid]["dtype"])

    def ceil(self, tid):
        return self._new(list(self.store[tid]["shape"]), [math.ceil(v) for v in self.store[tid]["values"]], self.store[tid]["dtype"])

    def round(self, tid):
        return self._new(list(self.store[tid]["shape"]), [round(v) for v in self.store[tid]["values"]], self.store[tid]["dtype"])

    def fill(self, tid, value):
        t = self.store[tid]
        t["values"] = [float(value)] * len(t["values"])
        return {"id": tid, "shape": list(t["shape"]), "dtype": str(t["dtype"])}

    def tril(self, tid, diagonal=0):
        t = self.store[tid]
        return self._new(list(t["shape"]), list(t["values"]), str(t["dtype"]))

    def triu(self, tid, diagonal=0):
        t = self.store[tid]
        return self._new(list(t["shape"]), list(t["values"]), str(t["dtype"]))

    def flip(self, tid, dims):
        t = self.store[tid]
        return self._new(list(t["shape"]), list(reversed(t["values"])), str(t["dtype"]))

    def sincos(self, tid):
        t = self.store[tid]
        return self._new(list(t["shape"]), [math.sin(v) for v in t["values"]], str(t["dtype"]))

    def dot(self, a, b):
        va = self.store[a]["values"]
        vb = self.store[b]["values"]
        return self._new([], [sum(x*y for x, y in zip(va, vb))], self.store[a]["dtype"])

    def arange(self, start, end, step, dtype):
        vals = []
        cur = float(start)
        while cur < float(end):
            vals.append(cur)
            cur += float(step)
        return self._new([len(vals)], vals, dtype)

    def full(self, shape, fill_value, dtype):
        return self._new(shape, [float(fill_value)] * math.prod(shape), dtype)

    def fullLike(self, tid, fill_value, dtype=None):
        t = self.store[tid]
        dt = str(t["dtype"]) if dtype is None else dtype
        return self._new(list(t["shape"]), [float(fill_value)] * math.prod(t["shape"]), dt)

    def rand(self, shape, dtype):
        return self._new(shape, [(i*1103515245+12345)%65536/65535.0 for i in range(math.prod(shape))], dtype)

    def randn(self, shape, dtype):
        return self._new(shape, [0.0]*math.prod(shape), dtype)

    def empty(self, shape, dtype):
        return self._new(shape, [0.0]*math.prod(shape), dtype)

    def emptyLike(self, tid, dtype=None):
        t = self.store[tid]
        dt = str(t["dtype"]) if dtype is None else dtype
        return self._new(list(t["shape"]), [0.0]*math.prod(t["shape"]), dt)

    def pow(self, a, b):
        va = self.store[a]["values"]
        vb = self.store[b]["values"]
        return self._new(list(self.store[a]["shape"]), [x**y for x, y in zip(va, vb)], self.store[a]["dtype"])

    def maximum(self, a, b):
        return self._binary(a, b, max)

    def minimum(self, a, b):
        return self._binary(a, b, min)

    def any(self, tid):
        v = self.store[tid]["values"]
        return self._new([], [1.0 if any(float(x)!=0 for x in v) else 0.0], self.store[tid]["dtype"])

    def all(self, tid):
        v = self.store[tid]["values"]
        return self._new([], [1.0 if all(float(x)!=0 for x in v) else 0.0], self.store[tid]["dtype"])

    def cumsum(self, tid):
        v = self.store[tid]["values"]
        acc = 0.0
        out = []
        for x in v:
            acc += x; out.append(acc)
        return self._new(list(self.store[tid]["shape"]), out, self.store[tid]["dtype"])

    def cumprod(self, tid):
        v = self.store[tid]["values"]
        acc = 1.0
        out = []
        for x in v:
            acc *= x; out.append(acc)
        return self._new(list(self.store[tid]["shape"]), out, self.store[tid]["dtype"])

    def heaviside(self, input_id, values_id):
        a = self.store[input_id]["values"]
        b = self.store[values_id]["values"]
        return self._new(list(self.store[input_id]["shape"]), [1.0 if v > 0 else (0.0 if v < 0 else y) for v, y in zip(a, b)], self.store[input_id]["dtype"])

    def sign(self, tid):
        return self._new(list(self.store[tid]["shape"]), [0.0 if v==0 else (1.0 if v>0 else -1.0) for v in self.store[tid]["values"]], self.store[tid]["dtype"])

    def _binary(self, a, b, fn):
        va = self.store[a]["values"]
        vb = self.store[b]["values"]
        return self._new(list(self.store[a]["shape"]), [fn(x, y) for x, y in zip(va, vb)], self.store[a]["dtype"])

    def indexSelect(self, tid, dim, index_id):
        t = self.store[tid]
        shape = list(t["shape"])
        d = dim if dim >= 0 else dim + len(shape)
        idx_vals = self.store[index_id]["values"]
        out_shape = list(shape)
        out_shape[d] = len(idx_vals)
        stride = 1
        for s in reversed(shape[d+1:]):
            stride *= s
        outer = 1
        for s in shape[:d]:
            outer *= s
        vals = t["values"]
        out = []
        for o in range(outer):
            for i in idx_vals:
                ii = int(i)
                for s in range(stride):
                    idx = o * shape[d] * stride + ii * stride + s
                    out.append(vals[idx])
        return self._new(out_shape, out, str(t["dtype"]))

    def maskedSelect(self, tid, mask_id):
        t = self.store[tid]
        mask_vals = self.store[mask_id]["values"]
        vals = t["values"]
        out = [v for v, m in zip(vals, mask_vals) if float(m) > 0]
        return self._new([len(out)], out, str(t["dtype"]))

    def sliceBackward(self, grad_output_id, input_shape, sliced_shape, dim, start, step):
        n = max(1, math.prod(input_shape))
        return self._new(list(input_shape), [0.0] * n, "float32")

    def cat(self, ids, dim):
        if len(ids) != 2:
            raise NotImplementedError("Only 2-tensor cat for fake runtime")
        a = self.store[ids[0]]
        b = self.store[ids[1]]
        a_shape = list(a["shape"])
        d = dim if dim >= 0 else dim + len(a_shape)
        out_shape = list(a_shape)
        out_shape[d] = a_shape[d] + list(b["shape"])[d]
        a_len = len(a["values"])
        out_vals = list(a["values"]) + list(b["values"])
        return self._new(out_shape, out_vals, str(a["dtype"]))


# ── Fixture ─────────────────────────────────────────────────────────

def fake_runtime(monkeypatch):
    fake = FakeRuntime()
    monkeypatch.setattr(tensor_mod, "_get_runtime", lambda: fake)
    monkeypatch.setattr(tensor_mod, "_run_js_awaitable", lambda value: value)
    return fake


# ── Testes individuais ──────────────────────────────────────────────

def test_grad_select(monkeypatch):
    """_grad_select deve colocar grad_output na posição [dim, index] do grad_input."""
    fake = fake_runtime(monkeypatch)
    from torch._tensor import tensor_from_data
    from torch.autograd import _grad_select

    # input = [[1, 2], [3, 4]]; select(dim= tys0, index=1) => [3, 4]
    inp = tensor_from_data([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    grad_out = tensor_from_data([10.0, 20.0], [2])  # grad for [3, 4]

    grad_inp = _grad_select(grad_out, inp, 0, 1)

    # Result should be zeros_like(inp) with [10, 20] at position [1, :]
    expected = [[0.0, 0.0], [10.0, 20.0]]
    assert grad_inp.tolist() == expected, f"Expected {expected}, got {grad_inp.tolist()}"
    print("✓ test_grad_select passed")


def test_grad_select_dim1(monkeypatch):
    """_grad_select ao longo da dim 1."""
    fake = fake_runtime(monkeypatch)
    from torch._tensor import tensor_from_data
    from torch.autograd import _grad_select

    inp = tensor_from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
    grad_out = tensor_from_data([100.0, 200.0], [2])  # select dim=1, index=0 => [1, 4]
    grad_inp = _grad_select(grad_out, inp, 1, 0)
    expected = [[100.0, 0.0, 0.0], [200.0, 0.0, 0.0]]
    assert grad_inp.tolist() == expected, f"Expected {expected}, got {grad_inp.tolist()}"
    print("✓ test_grad_select_dim1 passed")


def test_grad_max(monkeypatch):
    """_grad_max deve colocar grad_output na posição do argmax."""
    fake = fake_runtime(monkeypatch)
    from torch._tensor import tensor_from_data
    from torch.autograd import _grad_max

    inp = tensor_from_data([1.0, 5.0, 2.0, 3.0], [4], requires_grad=True)
    grad_out = tensor_from_data([42.0], [])
    grad_inp = _grad_max(grad_out, inp)
    # argmax is index 1 (value 5.0), so grad should be [0, 42, 0, 0]
    expected = [0.0, 42.0, 0.0, 0.0]
    assert grad_inp.tolist() == expected, f"Expected {expected}, got {grad_inp.tolist()}"
    print("✓ test_grad_max passed")


def test_grad_min(monkeypatch):
    """_grad_min deve colocar grad_output na posição do argmin."""
    fake = fake_runtime(monkeypatch)
    from torch._tensor import tensor_from_data
    from torch.autograd import _grad_min

    inp = tensor_from_data([3.0, 1.0, 4.0, 2.0], [4], requires_grad=True)
    grad_out = tensor_from_data([99.0], [])
    grad_inp = _grad_min(grad_out, inp)
    # argmin is index 1 (value 1.0), so grad should be [0, 99, 0, 0]
    expected = [0.0, 99.0, 0.0, 0.0]
    assert grad_inp.tolist() == expected, f"Expected {expected}, got {grad_inp.tolist()}"
    print("✓ test_grad_min passed")


def test_grad_masked_select(monkeypatch):
    """_grad_masked_select deve colocar grad_output nas posições True da mask."""
    fake = fake_runtime(monkeypatch)
    from torch._tensor import tensor_from_data
    from torch.autograd import _grad_masked_select

    inp = tensor_from_data([1.0, 2.0, 3.0, 4.0], [4], requires_grad=True)
    mask = tensor_from_data([1.0, 0.0, 1.0, 0.0], [4])  # bool mask
    grad_out = tensor_from_data([10.0, 30.0], [2])  # values at mask positions
    grad_inp = _grad_masked_select(grad_out, inp, mask)
    expected = [10.0, 0.0, 30.0, 0.0]
    assert grad_inp.tolist() == expected, f"Expected {expected}, got {grad_inp.tolist()}"
    print("✓ test_grad_masked_select passed")


def test_grad_masked_select_2d(monkeypatch):
    """_grad_masked_select 2D."""
    fake = fake_runtime(monkeypatch)
    from torch._tensor import tensor_from_data
    from torch.autograd import _grad_masked_select

    inp = tensor_from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
    mask = tensor_from_data([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2, 3])
    grad_out = tensor_from_data([100.0, 500.0], [2])
    grad_inp = _grad_masked_select(grad_out, inp, mask)
    expected = [[100.0, 0.0, 0.0], [0.0, 500.0, 0.0]]
    assert grad_inp.tolist() == expected, f"Expected {expected}, got {grad_inp.tolist()}"
    print("✓ test_grad_masked_select_2d passed")


def test_grad_index_select(monkeypatch):
    """_grad_index_select deve colocar grad_output nas posições dos índices."""
    fake = fake_runtime(monkeypatch)
    from torch._tensor import tensor_from_data
    from torch.autograd import _grad_index_select

    inp = tensor_from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], requires_grad=True)
    index = tensor_from_data([0.0, 2.0], [2], dtype="int32")  # select rows 0 and 2
    grad_out = tensor_from_data([10.0, 20.0, 100.0, 200.0], [2, 2])
    grad_inp = _grad_index_select(grad_out, inp, 0, index)
    # Row 0 gets [10, 20], Row 2 gets [100, 200], Row 1 stays [0, 0]
    expected = [[10.0, 20.0], [0.0, 0.0], [100.0, 200.0]]
    assert grad_inp.tolist() == expected, f"Expected {expected}, got {grad_inp.tolist()}"
    print("✓ test_grad_index_select passed")


def test_grad_index_select_dim1(monkeypatch):
    """_grad_index_select ao longo da dim 1."""
    fake = fake_runtime(monkeypatch)
    from torch._tensor import tensor_from_data
    from torch.autograd import _grad_index_select

    inp = tensor_from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
    index = tensor_from_data([0.0, 2.0], [2], dtype="int32")  # select cols 0 and 2
    grad_out = tensor_from_data([10.0, 30.0, 40.0, 60.0], [2, 2])
    grad_inp = _grad_index_select(grad_out, inp, 1, index)
    # Col 0 gets [10, 40], Col 2 gets [30, 60], Col 1 stays [0, 0]
    expected = [[10.0, 0.0, 30.0], [40.0, 0.0, 60.0]]
    assert grad_inp.tolist() == expected, f"Expected {expected}, got {grad_inp.tolist()}"
    print("✓ test_grad_index_select_dim1 passed")


def test_backward_select_forward(monkeypatch):
    """Teste integrado: tensor.select(dim, index).backward()"""
    fake = fake_runtime(monkeypatch)
    from torch import tensor

    x = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.select(0, 1)  # [3.0, 4.0]
    y.backward(tensor([10.0, 20.0]))
    expected = [[0.0, 0.0], [10.0, 20.0]]
    assert x.grad.tolist() == expected, f"Expected {expected}, got {x.grad.tolist()}"
    print("✓ test_backward_select_forward passed")


def test_backward_max_forward(monkeypatch):
    """Teste integrado: tensor.max().backward()"""
    fake = fake_runtime(monkeypatch)
    from torch import tensor

    x = tensor([3.0, 1.0, 4.0, 2.0], requires_grad=True)
    y = x.max()
    y.backward()
    # argmax is index 2 (value 4.0), so grad should be [0, 0, 1, 0]
    expected = [0.0, 0.0, 1.0, 0.0]
    assert x.grad.tolist() == expected, f"Expected {expected}, got {x.grad.tolist()}"
    print("✓ test_backward_max_forward passed")


def test_backward_min_forward(monkeypatch):
    """Teste integrado: tensor.min().backward()"""
    fake = fake_runtime(monkeypatch)
    from torch import tensor

    x = tensor([3.0, 1.0, 4.0, 2.0], requires_grad=True)
    y = x.min()
    y.backward()
    # argmin is index 1 (value 1.0), so grad should be [0, 1, 0, 0]
    expected = [0.0, 1.0, 0.0, 0.0]
    assert x.grad.tolist() == expected, f"Expected {expected}, got {x.grad.tolist()}"
    print("✓ test_backward_min_forward passed")


def test_backward_index_select_forward(monkeypatch):
    """Teste integrado: tensor.index_select(dim, index).backward()"""
    fake = fake_runtime(monkeypatch)
    from torch import tensor

    x = tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    idx = tensor([0, 2], dtype="int32")
    y = x.index_select(0, idx)
    y.backward(tensor([[10.0, 20.0], [50.0, 60.0]]))
    expected = [[10.0, 20.0], [0.0, 0.0], [50.0, 60.0]]
    assert x.grad.tolist() == expected, f"Expected {expected}, got {x.grad.tolist()}"
    print("✓ test_backward_index_select_forward passed")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
