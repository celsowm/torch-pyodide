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
    assert ("slice", (1, 0, 1, 1)) in t.calls
    assert ("index_select", (2, idx)) in t.calls


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
