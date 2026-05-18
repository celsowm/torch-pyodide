from __future__ import annotations

import math

import torch as torch_mod
import torch._tensor as tensor_mod


class FakeRuntime:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.initialized = False
        self.next_id = 100
        self.store: dict[int, dict[str, object]] = {}

    def _new(self, shape: list[int], values: list[float], dtype: str = "float32") -> dict[str, object]:
        tensor_id = self.next_id
        self.next_id += 1
        self.store[tensor_id] = {"shape": shape, "values": values, "dtype": dtype}
        return {"id": tensor_id, "shape": shape, "dtype": dtype}

    def init(self) -> None:
        if not self.available:
            raise RuntimeError("WebGPU unavailable in this browser.")
        self.initialized = True
        return None

    def _ensure_ready(self) -> None:
        if not self.available:
            raise RuntimeError("WebGPU unavailable in this browser.")
        if not self.initialized:
            self.initialized = True

    def isAvailable(self) -> bool:
        return self.available

    def isInitialized(self) -> bool:
        return self.initialized

    def deviceCount(self) -> int:
        return 1 if self.available else 0

    def currentDevice(self) -> int:
        self._ensure_ready()
        return 0

    def getDeviceName(self, idx: int = 0) -> str:
        if idx != 0:
            raise RuntimeError("Only device index 0 is supported in MVP")
        self._ensure_ready()
        return "Fake WebGPU Adapter"

    def getDeviceProperties(self, idx: int = 0) -> dict[str, object]:
        if idx != 0:
            raise RuntimeError("Only device index 0 is supported in MVP")
        self._ensure_ready()
        return {
            "name": "Fake WebGPU Adapter",
            "total_memory": 0,
            "major": 0,
            "minor": 0,
            "multi_processor_count": 0,
            "vendor": "fake-vendor",
            "architecture": "fake-arch",
            "description": "fake-description",
            "device": "fake-device",
            "is_fallback_adapter": False,
            "subgroup_min_size": 0,
            "subgroup_max_size": 0,
            "limits": {"maxBufferSize": 1024},
        }

    def memoryAllocated(self, idx: int = 0) -> int:
        if idx != 0:
            raise RuntimeError("Only device index 0 is supported in MVP")
        self._ensure_ready()
        total = 0
        for tensor in self.store.values():
            total += _numel(list(tensor["shape"])) * 4
        return total

    def memoryReserved(self, idx: int = 0) -> int:
        return self.memoryAllocated(idx)

    def tensorFromData(self, flat: list[float], shape: list[int], dtype: str) -> dict[str, object]:
        self._ensure_ready()
        return self._new(shape, flat, dtype)

    def zeros(self, shape: list[int], dtype: str) -> dict[str, object]:
        self._ensure_ready()
        size = _numel(shape)
        return self._new(shape, [0.0] * size, dtype)

    def ones(self, shape: list[int], dtype: str) -> dict[str, object]:
        self._ensure_ready()
        size = _numel(shape)
        return self._new(shape, [1.0] * size, dtype)

    def rand(self, shape: list[int], dtype: str) -> dict[str, object]:
        self._ensure_ready()
        size = _numel(shape)
        vals = [((i * 1103515245 + 12345) % 65536) / 65535.0 for i in range(size)]
        return self._new(shape, vals, dtype)

    def randn(self, shape: list[int], dtype: str) -> dict[str, object]:
        self._ensure_ready()
        size = _numel(shape)
        vals = [((i % 11) - 5) / 3.0 for i in range(size)]
        return self._new(shape, vals, dtype)

    def arange(self, start: float, end: float, step: float, dtype: str) -> dict[str, object]:
        self._ensure_ready()
        if step == 0:
            raise RuntimeError("arange step must be non-zero.")
        vals = []
        current = float(start)
        if step > 0:
            while current < float(end):
                vals.append(current)
                current += float(step)
        else:
            while current > float(end):
                vals.append(current)
                current += float(step)
        return self._new([len(vals)], vals, dtype)

    def full(self, shape: list[int], fill_value: float, dtype: str) -> dict[str, object]:
        self._ensure_ready()
        size = _numel(shape)
        return self._new(shape, [float(fill_value)] * size, dtype)

    def fullLike(self, tensor_id: int, fill_value: float, dtype: str | None = None) -> dict[str, object]:
        self._ensure_ready()
        t = self.store[tensor_id]
        out_dtype = str(t["dtype"]) if dtype is None else str(dtype)
        size = _numel(list(t["shape"]))
        return self._new(list(t["shape"]), [float(fill_value)] * size, out_dtype)

    def zerosLike(self, tensor_id: int, dtype: str | None = None) -> dict[str, object]:
        self._ensure_ready()
        t = self.store[tensor_id]
        out_dtype = str(t["dtype"]) if dtype is None else str(dtype)
        size = _numel(list(t["shape"]))
        return self._new(list(t["shape"]), [0.0] * size, out_dtype)

    def onesLike(self, tensor_id: int, dtype: str | None = None) -> dict[str, object]:
        self._ensure_ready()
        t = self.store[tensor_id]
        out_dtype = str(t["dtype"]) if dtype is None else str(dtype)
        size = _numel(list(t["shape"]))
        return self._new(list(t["shape"]), [1.0] * size, out_dtype)

    def add(self, a_id: int, b_id: int) -> dict[str, object]:
        return self._binary(a_id, b_id, lambda a, b: a + b)

    def sub(self, a_id: int, b_id: int) -> dict[str, object]:
        return self._binary(a_id, b_id, lambda a, b: a - b)

    def mul(self, a_id: int, b_id: int) -> dict[str, object]:
        return self._binary(a_id, b_id, lambda a, b: a * b)

    def div(self, a_id: int, b_id: int) -> dict[str, object]:
        return self._binary(a_id, b_id, lambda a, b: a / b)

    def _binary(self, a_id: int, b_id: int, fn) -> dict[str, object]:
        a = self.store[a_id]
        b = self.store[b_id]
        values = [fn(float(x), float(y)) for x, y in zip(a["values"], b["values"])]
        return self._new(list(a["shape"]), values, str(a["dtype"]))

    def matmul(self, a_id: int, b_id: int) -> dict[str, object]:
        a = self.store[a_id]
        b = self.store[b_id]
        a_shape = list(a["shape"])
        b_shape = list(b["shape"])
        m, k = int(a_shape[0]), int(a_shape[1])
        n = int(b_shape[1])
        out = []
        for i in range(m):
            for j in range(n):
                acc = 0.0
                for t in range(k):
                    acc += float(a["values"][i * k + t]) * float(b["values"][t * n + j])
                out.append(acc)
        return self._new([m, n], out, str(a["dtype"]))

    def sum(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        return self._new([], [sum(float(v) for v in t["values"])], str(t["dtype"]))

    def mean(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [float(v) for v in t["values"]]
        return self._new([], [sum(values) / len(values)], str(t["dtype"]))

    def relu(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [max(0.0, float(v)) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def abs(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [abs(float(v)) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def sqrt(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [float(v) ** 0.5 for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def exp(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [math.exp(float(v)) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def log(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [math.log(float(v)) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def neg(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [-float(v) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def clamp(self, tensor_id: int, min_val: float, max_val: float) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [max(min(float(v), max_val), min_val) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def where(self, cond_id: int, x_id: int, y_id: int) -> dict[str, object]:
        cond = self.store[cond_id]
        x = self.store[x_id]
        y = self.store[y_id]
        out = []
        for c, xv, yv in zip(cond["values"], x["values"], y["values"]):
            out.append(float(xv) if float(c) > 0.0 else float(yv))
        return self._new(list(x["shape"]), out, str(x["dtype"]))

    def argmax(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        shape = list(t["shape"])
        vals = [float(v) for v in t["values"]]
        if len(shape) == 1:
            idx = max(range(shape[0]), key=lambda i: vals[i]) if shape[0] > 0 else 0
            return self._new([], [float(idx)], "int32")
        last = int(shape[-1])
        batch = _numel(shape[:-1])
        out = []
        for b in range(batch):
            start = b * last
            idx = max(range(last), key=lambda i: vals[start + i])
            out.append(float(idx))
        return self._new(shape[:-1], out, "int32")

    def argmin(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        shape = list(t["shape"])
        vals = [float(v) for v in t["values"]]
        if len(shape) == 1:
            idx = min(range(shape[0]), key=lambda i: vals[i]) if shape[0] > 0 else 0
            return self._new([], [float(idx)], "int32")
        last = int(shape[-1])
        batch = _numel(shape[:-1])
        out = []
        for b in range(batch):
            start = b * last
            idx = min(range(last), key=lambda i: vals[start + i])
            out.append(float(idx))
        return self._new(shape[:-1], out, "int32")

    def reshape(self, tensor_id: int, shape: list[int]) -> dict[str, object]:
        t = self.store[tensor_id]
        return self._new(shape, list(t["values"]), str(t["dtype"]))

    def flatten(self, tensor_id: int, start_dim: int = 0, end_dim: int = -1) -> dict[str, object]:
        t = self.store[tensor_id]
        shape = list(t["shape"])
        rank = len(shape)
        if rank == 0:
            return self._new([1], list(t["values"]), str(t["dtype"]))
        start = start_dim if start_dim >= 0 else start_dim + rank
        end = end_dim if end_dim >= 0 else end_dim + rank
        if start < 0 or end < 0 or start >= rank or end >= rank or start > end:
            raise RuntimeError("invalid flatten dims")
        middle = _numel(shape[start : end + 1])
        out_shape = shape[:start] + [middle] + shape[end + 1 :]
        return self._new(out_shape, list(t["values"]), str(t["dtype"]))

    def squeeze(self, tensor_id: int, dim: int | None = None) -> dict[str, object]:
        t = self.store[tensor_id]
        shape = list(t["shape"])
        if dim is None:
            out_shape = [d for d in shape if d != 1]
        else:
            rank = len(shape)
            resolved = dim if dim >= 0 else dim + rank
            if resolved < 0 or resolved >= rank:
                raise RuntimeError("dim out of range")
            out_shape = list(shape)
            if out_shape[resolved] == 1:
                out_shape.pop(resolved)
        return self._new(out_shape, list(t["values"]), str(t["dtype"]))

    def unsqueeze(self, tensor_id: int, dim: int) -> dict[str, object]:
        t = self.store[tensor_id]
        shape = list(t["shape"])
        rank = len(shape)
        resolved = dim if dim >= 0 else dim + rank + 1
        if resolved < 0 or resolved > rank:
            raise RuntimeError("dim out of range")
        out_shape = list(shape)
        out_shape.insert(resolved, 1)
        return self._new(out_shape, list(t["values"]), str(t["dtype"]))

    def transpose(self, tensor_id: int, dim0: int, dim1: int) -> dict[str, object]:
        return self.permute(tensor_id, _swap_dims(list(range(len(self.store[tensor_id]["shape"]))), dim0, dim1))

    def permute(self, tensor_id: int, dims: list[int]) -> dict[str, object]:
        t = self.store[tensor_id]
        in_shape = list(t["shape"])
        rank = len(in_shape)
        norm = [d if d >= 0 else d + rank for d in dims]
        if len(norm) != rank or len(set(norm)) != rank:
            raise RuntimeError("invalid permute dims")
        out_shape = [in_shape[d] for d in norm]
        in_vals = [float(v) for v in t["values"]]
        in_strides = _strides(in_shape)
        out_strides = _strides(out_shape)
        out = [0.0] * len(in_vals)
        for out_idx in range(len(out)):
            out_coords = _coords(out_idx, out_shape, out_strides)
            in_coords = [0] * rank
            for i, axis in enumerate(norm):
                in_coords[axis] = out_coords[i]
            in_idx = sum(c * s for c, s in zip(in_coords, in_strides))
            out[out_idx] = in_vals[in_idx]
        return self._new(out_shape, out, str(t["dtype"]))

    def select(self, tensor_id: int, dim: int, index: int) -> dict[str, object]:
        t = self.store[tensor_id]
        in_shape = list(t["shape"])
        rank = len(in_shape)
        resolved_dim = dim if dim >= 0 else dim + rank
        axis = in_shape[resolved_dim]
        resolved_index = index if index >= 0 else index + axis
        out_shape = in_shape[:resolved_dim] + in_shape[resolved_dim + 1 :]
        out = []
        values = [float(v) for v in t["values"]]
        in_strides = _strides(in_shape)
        out_strides = _strides(out_shape)
        out_len = max(1, _numel(out_shape))
        for out_idx in range(out_len):
            out_coords = [] if len(out_shape) == 0 else _coords(out_idx, out_shape, out_strides)
            in_coords = out_coords[:resolved_dim] + [resolved_index] + out_coords[resolved_dim:]
            idx = sum(c * s for c, s in zip(in_coords, in_strides))
            out.append(values[idx])
        return self._new(out_shape, out, str(t["dtype"]))

    def slice(self, tensor_id: int, dim: int, start: int | None = None, end: int | None = None, step: int = 1) -> dict[str, object]:
        t = self.store[tensor_id]
        in_shape = list(t["shape"])
        rank = len(in_shape)
        resolved_dim = dim if dim >= 0 else dim + rank
        axis = in_shape[resolved_dim]
        rng = list(range(axis))[slice(start, end, step)]
        out_shape = list(in_shape)
        out_shape[resolved_dim] = len(rng)
        values = [float(v) for v in t["values"]]
        in_strides = _strides(in_shape)
        out_strides = _strides(out_shape)
        out_len = _numel(out_shape)
        out = [0.0] * out_len
        for out_idx in range(out_len):
            out_coords = _coords(out_idx, out_shape, out_strides)
            in_coords = list(out_coords)
            in_coords[resolved_dim] = rng[out_coords[resolved_dim]]
            idx = sum(c * s for c, s in zip(in_coords, in_strides))
            out[out_idx] = values[idx]
        return self._new(out_shape, out, str(t["dtype"]))

    def transpose2d(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        rows, cols = int(t["shape"][0]), int(t["shape"][1])
        vals = [float(v) for v in t["values"]]
        out = []
        for c in range(cols):
            for r in range(rows):
                out.append(vals[r * cols + c])
        return self._new([cols, rows], out, str(t["dtype"]))

    def toList(self, tensor_id: int) -> list[float]:
        t = self.store[tensor_id]
        if str(t["dtype"]) == "bool":
            return [1.0 if float(v) != 0.0 else 0.0 for v in t["values"]]
        if str(t["dtype"]) == "int32":
            return [float(int(v)) for v in t["values"]]
        return [float(v) for v in t["values"]]

    def destroy(self, tensor_id: int) -> None:
        self.store.pop(tensor_id, None)
        return None

    # ── Fase 0: unary activation ops ─────────────────────────────
    def sigmoid(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [1.0 / (1.0 + math.exp(-float(v))) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def tanh(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [math.tanh(float(v)) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def sin(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [math.sin(float(v)) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def cos(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [math.cos(float(v)) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def gelu(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [float(v) for v in t["values"]]
        out = []
        for x in values:
            cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
            out.append(x * cdf)
        return self._new(list(t["shape"]), out, str(t["dtype"]))

    def silu(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [float(v) / (1.0 + math.exp(-float(v))) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def leakyRelu(self, tensor_id: int, alpha: float = 0.01) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [max(alpha * float(v), float(v)) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def floor(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [math.floor(float(v)) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def ceil(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [math.ceil(float(v)) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def round(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [round(float(v)) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def reciprocal(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [1.0 / float(v) if float(v) != 0.0 else float("inf") for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def square(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [float(v) ** 2 for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    # ── Fase 0: unary trig ops ───────────────────────────────────
    def _unary(self, tensor_id: int, fn) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [fn(float(v)) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def tan(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, math.tan)

    def asin(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, math.asin)

    def acos(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, math.acos)

    def atan(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, math.atan)

    def sinh(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, math.sinh)

    def cosh(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, math.cosh)

    def asinh(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, math.asinh)

    def acosh(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, math.acosh)

    def atanh(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, math.atanh)

    def exp2(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, lambda x: 2.0 ** x)

    def log2(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, lambda x: math.log(x, 2))

    def log10(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, math.log10)

    def log1p(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, math.log1p)

    def expm1(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, math.expm1)

    def trunc(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, math.trunc)

    def frac(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, lambda x: x - math.floor(x))

    def softplus(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, lambda x: math.log(1.0 + math.exp(x)))

    def mish(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, lambda x: x * math.tanh(math.log(1.0 + math.exp(x))))

    def hardsigmoid(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, lambda x: max(0.0, min(1.0, x / 3.0 + 0.5)))

    def hardswish(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, lambda x: x * max(0.0, min(1.0, x / 3.0 + 0.5)))

    def softsign(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, lambda x: x / (1.0 + abs(x)))

    def tanhshrink(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, lambda x: x - math.tanh(x))

    def rsqrt(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, lambda x: x ** (-0.5) if x != 0.0 else float("inf"))

    def sign(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, lambda x: 0.0 if x == 0.0 else (1.0 if x > 0.0 else -1.0))

    def sgn(self, tensor_id: int) -> dict[str, object]:
        return self.sign(tensor_id)

    def _bool_unary(self, tensor_id: int, fn) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [1.0 if fn(float(v)) else 0.0 for v in t["values"]]
        return self._new(list(t["shape"]), values, "bool")

    def isnan(self, tensor_id: int) -> dict[str, object]:
        return self._bool_unary(tensor_id, math.isnan)

    def isinf(self, tensor_id: int) -> dict[str, object]:
        return self._bool_unary(tensor_id, math.isinf)

    def isfinite(self, tensor_id: int) -> dict[str, object]:
        return self._bool_unary(tensor_id, math.isfinite)

    def isposinf(self, tensor_id: int) -> dict[str, object]:
        return self._bool_unary(tensor_id, lambda x: x == float("inf"))

    def isneginf(self, tensor_id: int) -> dict[str, object]:
        return self._bool_unary(tensor_id, lambda x: x == float("-inf"))

    def logicalNot(self, tensor_id: int) -> dict[str, object]:
        return self._bool_unary(tensor_id, lambda x: x == 0.0)

    def erf(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, math.erf)

    def erfc(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, math.erfc)

    def lgamma(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, math.lgamma)

    def digamma(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, lambda x: x)

    def i0(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, lambda x: 1.0)

    def deg2rad(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, lambda x: x * math.pi / 180.0)

    def rad2deg(self, tensor_id: int) -> dict[str, object]:
        return self._unary(tensor_id, lambda x: x * 180.0 / math.pi)

    # ── Fase 1: missing ops ──────────────────────────────────────
    def pow(self, a_id: int, b_id: int) -> dict[str, object]:
        return self._binary(a_id, b_id, lambda a, b: a ** b)

    def heaviside(self, input_id: int, values_id: int) -> dict[str, object]:
        a = self.store[input_id]
        b = self.store[values_id]
        vals = [1.0 if float(x) > 0.0 else (0.0 if float(x) < 0.0 else float(y)) for x, y in zip(a["values"], b["values"])]
        return self._new(list(a["shape"]), vals, str(a["dtype"]))

    def maximum(self, a_id: int, b_id: int) -> dict[str, object]:
        return self._binary(a_id, b_id, max)

    def minimum(self, a_id: int, b_id: int) -> dict[str, object]:
        return self._binary(a_id, b_id, min)

    def any(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        any_val = 1.0 if any(float(v) != 0.0 for v in t["values"]) else 0.0
        return self._new([], [any_val], str(t["dtype"]))

    def all(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        all_val = 1.0 if all(float(v) != 0.0 for v in t["values"]) else 0.0
        return self._new([], [all_val], str(t["dtype"]))

    def cumsum(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        vals = [float(v) for v in t["values"]]
        acc = 0.0
        out = []
        for v in vals:
            acc += v
            out.append(acc)
        return self._new(list(t["shape"]), out, str(t["dtype"]))

    def cumprod(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        vals = [float(v) for v in t["values"]]
        acc = 1.0
        out = []
        for v in vals:
            acc *= v
            out.append(acc)
        return self._new(list(t["shape"]), out, str(t["dtype"]))

    def empty(self, shape: list[int], dtype: str) -> dict[str, object]:
        self._ensure_ready()
        size = _numel(shape)
        return self._new(shape, [0.0] * size, dtype)

    def emptyLike(self, tensor_id: int, dtype: str | None = None) -> dict[str, object]:
        self._ensure_ready()
        t = self.store[tensor_id]
        out_dtype = str(t["dtype"]) if dtype is None else str(dtype)
        size = _numel(list(t["shape"]))
        return self._new(list(t["shape"]), [0.0] * size, out_dtype)

    def tril(self, tensor_id: int, diagonal: int = 0) -> dict[str, object]:
        t = self.store[tensor_id]
        shape = list(t["shape"])
        vals = [float(v) for v in t["values"]]
        if len(shape) < 2:
            return self._new(shape, vals, str(t["dtype"]))
        rows, cols = shape[-2], shape[-1]
        stride = cols
        out = []
        for i in range(_numel(shape[:-2]) if len(shape) > 2 else 1):
            offset = i * rows * cols
            for r in range(rows):
                for c in range(cols):
                    if c <= r + diagonal:
                        out.append(vals[offset + r * stride + c])
                    else:
                        out.append(0.0)
        return self._new(shape, out, str(t["dtype"]))

    def triu(self, tensor_id: int, diagonal: int = 0) -> dict[str, object]:
        t = self.store[tensor_id]
        shape = list(t["shape"])
        vals = [float(v) for v in t["values"]]
        if len(shape) < 2:
            return self._new(shape, vals, str(t["dtype"]))
        rows, cols = shape[-2], shape[-1]
        stride = cols
        out = []
        for i in range(_numel(shape[:-2]) if len(shape) > 2 else 1):
            offset = i * rows * cols
            for r in range(rows):
                for c in range(cols):
                    if c >= r + diagonal:
                        out.append(vals[offset + r * stride + c])
                    else:
                        out.append(0.0)
        return self._new(shape, out, str(t["dtype"]))

    def flip(self, tensor_id: int, dims: list[int]) -> dict[str, object]:
        t = self.store[tensor_id]
        shape = list(t["shape"])
        vals = [float(v) for v in t["values"]]
        out = list(vals)
        for d in dims:
            dim = d if d >= 0 else d + len(shape)
            out = self._flip_dim(out, shape, dim)
        return self._new(shape, out, str(t["dtype"]))

    def _flip_dim(self, vals: list[float], shape: list[int], dim: int) -> list[float]:
        strides = _strides(shape)
        total = len(vals)
        out = [0.0] * total
        for i in range(total):
            coords = _coords(i, shape, strides)
            coords[dim] = shape[dim] - 1 - coords[dim]
            idx = sum(c * s for c, s in zip(coords, strides))
            out[i] = vals[idx]
        return out


def _numel(shape: list[int]) -> int:
    size = 1
    for d in shape:
        size *= int(d)
    return size


def _strides(shape: list[int]) -> list[int]:
    if len(shape) == 0:
        return []
    out = [0] * len(shape)
    running = 1
    for i in range(len(shape) - 1, -1, -1):
        out[i] = running
        running *= int(shape[i])
    return out


def _coords(index: int, shape: list[int], strides: list[int]) -> list[int]:
    rem = index
    coords: list[int] = []
    for i in range(len(shape)):
        stride = strides[i]
        coords.append(rem // stride)
        rem %= stride
    return coords


def _swap_dims(dims: list[int], d0: int, d1: int) -> list[int]:
    n = len(dims)
    r0 = d0 if d0 >= 0 else d0 + n
    r1 = d1 if d1 >= 0 else d1 + n
    dims[r0], dims[r1] = dims[r1], dims[r0]
    return dims


def test_torch_public_contract(monkeypatch):
    fake = FakeRuntime()
    monkeypatch.setattr(tensor_mod, "_get_runtime", lambda: fake)
    monkeypatch.setattr(tensor_mod, "_run_js_awaitable", lambda value: value)
    monkeypatch.setattr(torch_mod.cuda, "_get_runtime", lambda: fake)
    monkeypatch.setattr(torch_mod.cuda, "_run_js_awaitable", lambda value: value)

    a = torch_mod.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch_mod.ones((2, 2))

    c = torch_mod.add(a, b)
    d = torch_mod.mul(c, torch_mod.tensor([[2.0, 2.0], [2.0, 2.0]]))
    e = torch_mod.sub(d, torch_mod.ones((2, 2)))
    f = torch_mod.div(e, torch_mod.tensor([[1.0, 5.0], [1.0, 3.0]]))
    g = torch_mod.relu(f)
    h = torch_mod.matmul(a, torch_mod.tensor([[1.0, 0.0], [0.0, 1.0]]))
    clamped = torch_mod.clamp(torch_mod.tensor([[-1.0, 0.5], [2.5, 0.2]]), 0.0, 1.0)
    chosen = torch_mod.where(torch_mod.tensor([[1.0, 0.0], [0.0, 2.0]]), a, b)
    ix_max = torch_mod.argmax(torch_mod.tensor([[1.0, 3.0], [5.0, 4.0]]))
    ix_min = torch_mod.argmin(torch_mod.tensor([[1.0, 3.0], [5.0, 4.0]]))
    rnd = torch_mod.rand((2, 2))
    rndn = torch_mod.randn((2, 2))
    seq = torch_mod.arange(1, 6, 2, dtype="int32")
    filled = torch_mod.full((2, 2), 7.0, dtype="int32")
    filled_like = torch_mod.full_like(a, 9.0)
    zeros_like = torch_mod.zeros_like(a)
    ones_like = torch_mod.ones_like(a)
    abs_v = torch_mod.abs(torch_mod.tensor([[-1.0, 2.0], [-3.0, 4.0]]))
    sqrt_v = torch_mod.sqrt(torch_mod.tensor([[1.0, 4.0], [9.0, 16.0]]))
    exp_v = torch_mod.exp(torch_mod.tensor([[0.0, 1.0], [2.0, 0.0]]))
    log_v = torch_mod.log(torch_mod.tensor([[1.0, 2.718281828], [7.389056099, 1.0]]))
    neg_v = torch_mod.neg(torch_mod.tensor([[1.0, -2.0], [3.0, -4.0]]))
    bool_t = torch_mod.tensor([[1.0, 0.0], [0.0, 1.0]], dtype="bool")

    assert c.tolist() == [[2.0, 3.0], [4.0, 5.0]]
    assert d.tolist() == [[4.0, 6.0], [8.0, 10.0]]
    assert e.tolist() == [[3.0, 5.0], [7.0, 9.0]]
    assert f.tolist() == [[3.0, 1.0], [7.0, 3.0]]
    assert g.tolist() == [[3.0, 1.0], [7.0, 3.0]]
    assert h.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert clamped.tolist() == [[0.0, 0.5], [1.0, 0.2]]
    assert chosen.tolist() == [[1.0, 1.0], [1.0, 4.0]]
    assert ix_max.tolist() == [1.0, 0.0]
    assert ix_min.tolist() == [0.0, 1.0]
    assert tuple(rnd.shape) == (2, 2)
    assert all(0.0 <= float(v) <= 1.0 for row in rnd.tolist() for v in row)
    assert tuple(rndn.shape) == (2, 2)
    assert seq.tolist() == [1, 3, 5]
    assert seq.dtype == "int32"
    assert filled.tolist() == [[7, 7], [7, 7]]
    assert filled_like.tolist() == [[9.0, 9.0], [9.0, 9.0]]
    assert zeros_like.tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert ones_like.tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert abs_v.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert sqrt_v.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert neg_v.tolist() == [[-1.0, 2.0], [-3.0, 4.0]]
    assert bool_t.tolist() == [[True, False], [False, True]]
    assert abs(float(exp_v.tolist()[0][1]) - 2.718281828) < 1e-6
    assert abs(float(log_v.tolist()[0][1]) - 1.0) < 1e-5
    assert d.sum().tolist() == 28.0
    assert d.mean().tolist() == 7.0
    assert g.reshape((4,)).tolist() == [3.0, 1.0, 7.0, 3.0]
    assert g.T.tolist() == [[3.0, 7.0], [1.0, 3.0]]
    assert torch_mod.reshape(g, (4,)).tolist() == [3.0, 1.0, 7.0, 3.0]
    assert torch_mod.flatten(torch_mod.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]), 1, 2).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert torch_mod.squeeze(torch_mod.tensor([[[1.0, 2.0]]])).shape == (2,)
    assert torch_mod.unsqueeze(torch_mod.tensor([1.0, 2.0]), 0).shape == (1, 2)
    assert torch_mod.transpose(torch_mod.tensor([[1.0, 2.0], [3.0, 4.0]]), 0, 1).tolist() == [[1.0, 3.0], [2.0, 4.0]]
    assert torch_mod.permute(torch_mod.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]), (2, 0, 1)).shape == (2, 2, 1)
    assert torch_mod.select(torch_mod.tensor([[1.0, 2.0], [3.0, 4.0]]), 0, 1).tolist() == [3.0, 4.0]
    assert torch_mod.slice(torch_mod.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), 0, 0, 3, 2).tolist() == [[1.0, 2.0], [5.0, 6.0]]
    assert torch_mod.tensor([[1.0, 2.0], [3.0, 4.0]])[1].tolist() == [3.0, 4.0]
    assert torch_mod.tensor([1.0, 2.0, 3.0, 4.0])[1:4:2].tolist() == [2.0, 4.0]
    assert tuple(a.shape) == (2, 2)
    assert a.dtype == "float32"

    doomed = torch_mod.zeros((1,))
    doomed_id = doomed._id
    doomed.destroy()
    assert doomed_id not in fake.store
    assert torch_mod.cuda.is_available() is True
    assert torch_mod.cuda.device_count() == 1
    assert torch_mod.cuda.current_device() == 0
    assert torch_mod.cuda.get_device_name(0) == "Fake WebGPU Adapter"
    props = torch_mod.cuda.get_device_properties(0)
    assert props.total_memory == 0
    assert props.major == 0
    assert props.minor == 0
    assert props.multi_processor_count == 0
    assert props.vendor == "fake-vendor"
    assert isinstance(props.limits, dict)
    assert torch_mod.cuda.memory_allocated(0) == torch_mod.cuda.memory_reserved(0)
    assert torch_mod.cuda.memory.memory_allocated(0) == torch_mod.cuda.memory_allocated(0)
    assert torch_mod.cuda.memory.memory_reserved(0) == torch_mod.cuda.memory_reserved(0)
    try:
        torch_mod.arange(0, 10, 0)
        assert False, "expected arange step error"
    except RuntimeError as exc:
        assert "step must be non-zero" in str(exc)

    # ── New API tests ────────────────────────────────────────────
    empty_ = torch_mod.empty((2, 3))
    assert list(empty_.shape) == [2, 3]
    empty_like_ = torch_mod.empty_like(a)
    assert list(empty_like_.shape) == [2, 2]

    pow_v = torch_mod.pow(torch_mod.tensor([1.0, 2.0, 3.0]), torch_mod.tensor([2.0, 2.0, 2.0]))
    assert pow_v.tolist() == [1.0, 4.0, 9.0]

    max_v = torch_mod.maximum(torch_mod.tensor([1.0, 5.0, 3.0]), torch_mod.tensor([4.0, 2.0, 6.0]))
    assert max_v.tolist() == [4.0, 5.0, 6.0]

    min_v = torch_mod.minimum(torch_mod.tensor([1.0, 5.0, 3.0]), torch_mod.tensor([4.0, 2.0, 6.0]))
    assert min_v.tolist() == [1.0, 2.0, 3.0]

    h = torch_mod.heaviside(torch_mod.tensor([-1.0, 0.0, 1.0]), torch_mod.tensor([0.5, 0.5, 0.5]))
    assert h.tolist() == [0.0, 0.5, 1.0]

    any_v = torch_mod.any(torch_mod.tensor([0.0, 0.0, 1.0]))
    all_v = torch_mod.all(torch_mod.tensor([1.0, 1.0, 1.0]))
    assert any_v.tolist() == 1.0
    assert all_v.tolist() == 1.0

    cs = torch_mod.cumsum(torch_mod.tensor([1.0, 2.0, 3.0]))
    assert cs.tolist() == [1.0, 3.0, 6.0]

    cp = torch_mod.cumprod(torch_mod.tensor([1.0, 2.0, 3.0]))
    assert cp.tolist() == [1.0, 2.0, 6.0]

    tril_m = torch_mod.tril(torch_mod.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
    assert tril_m.tolist() == [[1.0, 0.0, 0.0], [4.0, 5.0, 0.0], [7.0, 8.0, 9.0]]

    triu_m = torch_mod.triu(torch_mod.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
    assert triu_m.tolist() == [[1.0, 2.0, 3.0], [0.0, 5.0, 6.0], [0.0, 0.0, 9.0]]

    flip_v = torch_mod.flip(torch_mod.tensor([1.0, 2.0, 3.0]), [0])
    assert flip_v.tolist() == [3.0, 2.0, 1.0]

    # Test a few new unary ops
    tan_v = torch_mod.tan(torch_mod.tensor([0.0, 0.7853981633974483]))
    assert abs(float(tan_v.tolist()[0])) < 1e-6

    s = torch_mod.sign(torch_mod.tensor([-2.0, 0.0, 3.0]))
    assert s.tolist() == [-1.0, 0.0, 1.0]

    s2 = torch_mod.sgn(torch_mod.tensor([-2.0, 0.0, 3.0]))
    assert s2.tolist() == [-1.0, 0.0, 1.0]

    n = torch_mod.isnan(torch_mod.tensor([1.0, float("nan")]))
    assert n.tolist() == [False, True]

    inf_ = torch_mod.isinf(torch_mod.tensor([1.0, float("inf")]))
    assert inf_.tolist() == [False, True]

    not_ = torch_mod.logical_not(torch_mod.tensor([1.0, 0.0, 1.0]))
    assert not_.tolist() == [False, True, False]

    log2_v = torch_mod.log2(torch_mod.tensor([1.0, 2.0, 4.0]))
    assert abs(float(log2_v.tolist()[2]) - 2.0) < 1e-5


def test_torch_cuda_unavailable_contract(monkeypatch):
    fake = FakeRuntime(available=False)
    monkeypatch.setattr(tensor_mod, "_get_runtime", lambda: fake)
    monkeypatch.setattr(tensor_mod, "_run_js_awaitable", lambda value: value)
    monkeypatch.setattr(torch_mod.cuda, "_get_runtime", lambda: fake)
    monkeypatch.setattr(torch_mod.cuda, "_run_js_awaitable", lambda value: value)

    assert torch_mod.cuda.is_available() is False
    assert torch_mod.cuda.device_count() == 0
    for fn in (
        torch_mod.cuda.current_device,
        lambda: torch_mod.cuda.get_device_name(0),
        lambda: torch_mod.cuda.get_device_properties(0),
        lambda: torch_mod.cuda.memory_allocated(0),
        lambda: torch_mod.cuda.memory_reserved(0),
    ):
        try:
            fn()
            assert False, "expected unavailable error"
        except RuntimeError as exc:
            assert "WebGPU unavailable" in str(exc)
