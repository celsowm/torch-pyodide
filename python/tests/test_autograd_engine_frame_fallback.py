from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.autograd_engine import _Node, _backward_from_tensor


class _FakeGrad:
    def __init__(self, value: float) -> None:
        self.value = value

    def add(self, other: "_FakeGrad") -> "_FakeGrad":
        return _FakeGrad(self.value + other.value)


class _FakeTensor:
    def __init__(self, requires_grad: bool = True) -> None:
        self._requires_grad = requires_grad
        self._shape = [1]
        self._dtype = "float32"
        self._node = None
        self.grad = None
        self._backward_hooks = {}


def _make_unary_graph() -> tuple[_FakeTensor, _FakeTensor]:
    x = _FakeTensor(requires_grad=True)
    y = _FakeTensor(requires_grad=True)
    y._node = _Node(y, lambda g: (g,), [x])
    return x, y


def test_backward_runs_when_runtime_frame_unavailable(monkeypatch):
    monkeypatch.setattr("torch._runtime._get_runtime", lambda: (_ for _ in ()).throw(RuntimeError("no runtime frame API")))

    x, y = _make_unary_graph()
    _backward_from_tensor(y, gradient=_FakeGrad(1.0))
    assert x.grad is not None
    assert x.grad.value == 1.0


@pytest.mark.parametrize(
    "name",
    ["sigmoid", "tanh", "gelu", "silu", "leaky_relu", "softmax"],
)
def test_activation_like_backward_populates_leaf_grad_even_without_frame(monkeypatch, name: str):
    monkeypatch.setattr("torch._runtime._get_runtime", lambda: (_ for _ in ()).throw(RuntimeError("no runtime frame API")))

    x, y = _make_unary_graph()
    _backward_from_tensor(y, gradient=_FakeGrad(1.0))
    assert x.grad is not None, f"{name} should populate leaf grad"


def test_topk_like_backward_populates_leaf_grad_even_without_frame(monkeypatch):
    monkeypatch.setattr("torch._runtime._get_runtime", lambda: (_ for _ in ()).throw(RuntimeError("no runtime frame API")))

    x = _FakeTensor(requires_grad=True)
    values_topk = _FakeTensor(requires_grad=True)
    values_topk._node = _Node(values_topk, lambda g: (g,), [x])

    _backward_from_tensor(values_topk, gradient=_FakeGrad(1.0))
    assert x.grad is not None
    assert x.grad.value == 1.0


def test_backward_does_not_suppress_grad_fn_errors(monkeypatch):
    monkeypatch.setattr("torch._runtime._get_runtime", lambda: (_ for _ in ()).throw(RuntimeError("no runtime frame API")))

    x = _FakeTensor(requires_grad=True)
    y = _FakeTensor(requires_grad=True)

    def _raise(_: _FakeGrad):
        raise ValueError("grad_fn exploded")

    y._node = _Node(y, _raise, [x])

    with pytest.raises(ValueError, match="grad_fn exploded"):
        _backward_from_tensor(y, gradient=_FakeGrad(1.0))
