from __future__ import annotations

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
        import math

        t = self.store[tensor_id]
        values = [math.exp(float(v)) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

    def log(self, tensor_id: int) -> dict[str, object]:
        import math

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


def _numel(shape: list[int]) -> int:
    size = 1
    for d in shape:
        size *= int(d)
    return size


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
    assert g.to_list() == g.tolist()
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
