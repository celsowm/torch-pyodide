from __future__ import annotations

import torch as torch_mod
import torch._tensor as tensor_mod


class FakeRuntime:
    def __init__(self) -> None:
        self.next_id = 100
        self.store: dict[int, dict[str, object]] = {}

    def _new(self, shape: list[int], values: list[float], dtype: str = "float32") -> dict[str, object]:
        tensor_id = self.next_id
        self.next_id += 1
        self.store[tensor_id] = {"shape": shape, "values": values, "dtype": dtype}
        return {"id": tensor_id, "shape": shape, "dtype": dtype}

    def init(self) -> None:
        return None

    def tensorFromData(self, flat: list[float], shape: list[int], dtype: str) -> dict[str, object]:
        return self._new(shape, flat, dtype)

    def zeros(self, shape: list[int], dtype: str) -> dict[str, object]:
        size = _numel(shape)
        return self._new(shape, [0.0] * size, dtype)

    def ones(self, shape: list[int], dtype: str) -> dict[str, object]:
        size = _numel(shape)
        return self._new(shape, [1.0] * size, dtype)

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
        return self._new([1], [sum(float(v) for v in t["values"])], str(t["dtype"]))

    def mean(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [float(v) for v in t["values"]]
        return self._new([1], [sum(values) / len(values)], str(t["dtype"]))

    def relu(self, tensor_id: int) -> dict[str, object]:
        t = self.store[tensor_id]
        values = [max(0.0, float(v)) for v in t["values"]]
        return self._new(list(t["shape"]), values, str(t["dtype"]))

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
        return [float(v) for v in self.store[tensor_id]["values"]]

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
    monkeypatch.setattr(torch_mod, "_get_runtime", lambda: fake)
    monkeypatch.setattr(torch_mod, "_run_js_awaitable", lambda value: value)

    torch_mod.init()
    a = torch_mod.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch_mod.ones((2, 2))

    c = torch_mod.add(a, b)
    d = torch_mod.mul(c, torch_mod.tensor([[2.0, 2.0], [2.0, 2.0]]))
    e = torch_mod.sub(d, torch_mod.ones((2, 2)))
    f = torch_mod.div(e, torch_mod.tensor([[1.0, 5.0], [1.0, 3.0]]))
    g = torch_mod.relu(f)
    h = torch_mod.matmul(a, torch_mod.tensor([[1.0, 0.0], [0.0, 1.0]]))

    assert c.to_list() == [2.0, 3.0, 4.0, 5.0]
    assert d.to_list() == [4.0, 6.0, 8.0, 10.0]
    assert e.to_list() == [3.0, 5.0, 7.0, 9.0]
    assert f.to_list() == [3.0, 1.0, 7.0, 3.0]
    assert g.to_list() == [3.0, 1.0, 7.0, 3.0]
    assert h.to_list() == [1.0, 2.0, 3.0, 4.0]
    assert d.sum().to_list() == [28.0]
    assert d.mean().to_list() == [7.0]
    assert g.reshape((4,)).to_list() == [3.0, 1.0, 7.0, 3.0]
    assert g.T.to_list() == [3.0, 7.0, 1.0, 3.0]
    assert tuple(a.shape) == (2, 2)
    assert a.dtype == "float32"

    doomed = torch_mod.zeros((1,))
    doomed_id = doomed._id
    doomed.destroy()
    assert doomed_id not in fake.store
