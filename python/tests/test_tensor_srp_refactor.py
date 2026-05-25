from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch._tensor import Tensor
from torch._tensor_indexing import getitem_from_tensor


class _FakeTensor:
    def __init__(self, ndim: int = 2, dtype: str = "float32") -> None:
        self.ndim = ndim
        self._dtype = dtype
        self._shape = [2, 2]
        self.calls: list[tuple[str, object]] = []

    def select(self, dim: int, index: int):
        self.calls.append(("select", (dim, index)))
        return self

    def slice(self, dim: int, start: int | None = None, end: int | None = None, step: int = 1):
        self.calls.append(("slice", (dim, start, end, step)))
        return self

    def index_select(self, dim: int, index: "_FakeTensor"):
        self.calls.append(("index_select", (dim, index)))
        return self

    def flatten(self):
        self.calls.append(("flatten", None))
        return self

    def item(self) -> float:
        return 0.0


def test_getitem_int_uses_select():
    t = _FakeTensor()
    out = getitem_from_tensor(t, 1)
    assert out is t
    assert t.calls == [("select", (0, 1))]


def test_getitem_slice_uses_slice():
    t = _FakeTensor()
    out = getitem_from_tensor(t, slice(0, 2, 1))
    assert out is t
    assert t.calls == [("slice", (0, 0, 2, 1))]


def test_getitem_tuple_mixed_uses_select_slice_and_index_select():
    t = _FakeTensor(ndim=2)
    idx = _FakeTensor(ndim=1, dtype="int32")
    out = getitem_from_tensor(t, (1, slice(0, 1, None), idx))
    assert out is t
    assert ("select", (0, 1)) in t.calls
    assert ("slice", (0, 0, 1, 1)) in t.calls
    assert ("index_select", (1, idx)) in t.calls


def test_getitem_invalid_type_raises():
    t = _FakeTensor()
    try:
        getitem_from_tensor(t, 1.5)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError for unsupported index type")


def test_tensor_linalg_methods_delegate(monkeypatch):
    called = {"det": False, "inv": False, "diag": False}
    sentinel = Tensor(999, [1], "float32")

    def _det(t: Tensor) -> Tensor:
        called["det"] = True
        return sentinel

    def _inv(t: Tensor) -> Tensor:
        called["inv"] = True
        return sentinel

    def _diag(t: Tensor) -> Tensor:
        called["diag"] = True
        return sentinel

    monkeypatch.setattr("torch._tensor_linalg_py.det_from_tensor", _det)
    monkeypatch.setattr("torch._tensor_linalg_py.inv_from_tensor", _inv)
    monkeypatch.setattr("torch._tensor_linalg_py.diag_from_tensor", _diag)

    t = Tensor(1, [2, 2], "float32")
    assert t.det() is sentinel
    assert t.inv() is sentinel
    assert t.diag() is sentinel
    assert called == {"det": True, "inv": True, "diag": True}


def test_tensor_runtime_bridge_methods_delegate(monkeypatch):
    called = {
        "item": False,
        "clamp": False,
        "argmax": False,
        "argmin": False,
        "tolist": False,
        "t": False,
        "destroy": False,
        "cholesky": False,
        "lu": False,
        "triangular_solve": False,
        "repeat": False,
        "reshape": False,
        "flatten": False,
        "squeeze": False,
        "unsqueeze": False,
        "transpose": False,
        "permute": False,
    }
    sentinel = object()

    monkeypatch.setattr("torch._tensor_runtime_bridge.item_from_tensor", lambda t: called.__setitem__("item", True) or 1.0)
    monkeypatch.setattr("torch._tensor_runtime_bridge.clamp_from_tensor", lambda t, mn, mx: called.__setitem__("clamp", True) or sentinel)
    monkeypatch.setattr("torch._tensor_runtime_bridge.argmax_from_tensor", lambda t: called.__setitem__("argmax", True) or sentinel)
    monkeypatch.setattr("torch._tensor_runtime_bridge.argmin_from_tensor", lambda t: called.__setitem__("argmin", True) or sentinel)
    monkeypatch.setattr("torch._tensor_runtime_bridge.tolist_from_tensor", lambda t: called.__setitem__("tolist", True) or [1.0])
    monkeypatch.setattr("torch._tensor_runtime_bridge.t_from_tensor", lambda t: called.__setitem__("t", True) or sentinel)
    monkeypatch.setattr("torch._tensor_runtime_bridge.destroy_tensor", lambda t: called.__setitem__("destroy", True) or None)
    monkeypatch.setattr("torch._tensor_runtime_bridge.cholesky_from_tensor", lambda t: called.__setitem__("cholesky", True) or sentinel)
    monkeypatch.setattr("torch._tensor_runtime_bridge.lu_from_tensor", lambda t: called.__setitem__("lu", True) or (sentinel, sentinel))
    monkeypatch.setattr("torch._tensor_runtime_bridge.triangular_solve_from_tensors", lambda a, b, up=False: called.__setitem__("triangular_solve", True) or sentinel)
    monkeypatch.setattr("torch._tensor_runtime_bridge.repeat_from_tensor", lambda t, sizes: called.__setitem__("repeat", True) or sentinel)
    monkeypatch.setattr("torch._tensor_runtime_bridge.reshape_from_tensor", lambda t, shape: called.__setitem__("reshape", True) or sentinel)
    monkeypatch.setattr("torch._tensor_runtime_bridge.flatten_from_tensor", lambda t, sd=0, ed=-1: called.__setitem__("flatten", True) or sentinel)
    monkeypatch.setattr("torch._tensor_runtime_bridge.squeeze_from_tensor", lambda t, d=None: called.__setitem__("squeeze", True) or sentinel)
    monkeypatch.setattr("torch._tensor_runtime_bridge.unsqueeze_from_tensor", lambda t, d: called.__setitem__("unsqueeze", True) or sentinel)
    monkeypatch.setattr("torch._tensor_runtime_bridge.transpose_from_tensor", lambda t, d0, d1: called.__setitem__("transpose", True) or sentinel)
    monkeypatch.setattr("torch._tensor_runtime_bridge.permute_from_tensor", lambda t, dims: called.__setitem__("permute", True) or sentinel)

    t = Tensor(1, [2, 2], "float32")
    assert t.item() == 1.0
    assert t.clamp(0.0, 1.0) is sentinel
    assert t.argmax() is sentinel
    assert t.argmin() is sentinel
    assert t.tolist() == [1.0]
    assert t.T is sentinel
    t.destroy()
    assert t.cholesky() is sentinel
    assert t.lu() == (sentinel, sentinel)
    assert t.triangular_solve(t) is sentinel
    assert t.repeat(2, 3) is sentinel
    assert t.reshape([4, 1]) is sentinel
    assert t.flatten() is sentinel
    assert t.squeeze() is sentinel
    assert t.unsqueeze(0) is sentinel
    assert t.transpose(0, 1) is sentinel
    assert t.permute((1, 0)) is sentinel
    assert all(called.values())


def test_tensor_shape_ops_delegate(monkeypatch):
    called = {"split": False, "chunk": False}
    sentinel_split = [object()]
    sentinel_chunk = [object(), object()]

    monkeypatch.setattr("torch._tensor_shape_ops.split_from_tensor", lambda t, s, d=0: called.__setitem__("split", True) or sentinel_split)
    monkeypatch.setattr("torch._tensor_shape_ops.chunk_from_tensor", lambda t, c, d=0: called.__setitem__("chunk", True) or sentinel_chunk)

    t = Tensor(1, [4, 4], "float32")
    assert t.split(2) is sentinel_split
    assert t.chunk(2) is sentinel_chunk
    assert all(called.values())


def test_tensor_math_helpers_delegate(monkeypatch):
    called = {"repeat_interleave": False, "norm": False, "radd": False, "rsub": False, "invert": False}
    sentinel = object()
    t = Tensor(1, [4, 4], "float32")

    monkeypatch.setattr("torch._tensor_math_helpers.repeat_interleave_from_tensor", lambda x, r, d=None: called.__setitem__("repeat_interleave", True) or sentinel)
    monkeypatch.setattr("torch._tensor_math_helpers.norm_from_tensor", lambda x, p="fro": called.__setitem__("norm", True) or sentinel)
    monkeypatch.setattr("torch._tensor_math_helpers.radd_from_tensor", lambda x, other: called.__setitem__("radd", True) or sentinel)
    monkeypatch.setattr("torch._tensor_math_helpers.rsub_from_tensor", lambda x, other: called.__setitem__("rsub", True) or sentinel)
    monkeypatch.setattr("torch._tensor_math_helpers.invert_from_tensor", lambda x: called.__setitem__("invert", True) or sentinel)

    assert t.repeat_interleave(2) is sentinel
    assert t.norm() is sentinel
    assert t.__radd__(2.0) is sentinel
    assert t.__rsub__(2.0) is sentinel
    assert t.__invert__() is sentinel
    assert all(called.values())
