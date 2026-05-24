from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.autograd_engine import _Node, _backward_from_tensor
from torch import autograd_rules

class _FakeGrad:
    def __init__(self, value: float | list, shape: list[int] | None = None) -> None:
        self.value = value
        if shape is not None:
            self._shape = shape
        else:
            self._shape = [len(value)] if isinstance(value, list) else [1]

    def add(self, other: _FakeGrad) -> _FakeGrad:
        if isinstance(self.value, list) and isinstance(other.value, list):
            return _FakeGrad([s + o for s, o in zip(self.value, other.value)], self._shape)
        return _FakeGrad(self.value + other.value, self._shape)

    def neg(self) -> _FakeGrad:
        if isinstance(self.value, list):
            return _FakeGrad([-v for v in self.value], self._shape)
        return _FakeGrad(-self.value, self._shape)

    def tolist(self) -> list:
        return self.value if isinstance(self.value, list) else [self.value]


class _FakeTensor:
    def __init__(self, shape: list[int], requires_grad: bool = True, dtype: str = "float32", data: list | None = None) -> None:
        self._requires_grad = requires_grad
        self._shape = shape
        self._dtype = dtype
        self.dtype = dtype
        self._node = None
        self.grad = None
        self._backward_hooks = {}
        self._data = data

    def tolist(self) -> list:
        if self._data is not None:
            return self._data
        if len(self._shape) == 1:
            return [1.0] * self._shape[0]
        elif len(self._shape) == 2:
            return [[1.0] * self._shape[1]] * self._shape[0]
        return [1.0]

    def to(self, dtype: str) -> _FakeTensor:
        return self


def test_grad_maximum_alignment():
    a = _FakeTensor([1], requires_grad=False)
    b = _FakeTensor([1], requires_grad=True)
    
    parents = [a, b]
    res = _FakeTensor([1], requires_grad=True)
    
    a.ge = lambda other: _FakeTensor([1], requires_grad=False)
    b.gt = lambda other: _FakeTensor([1], requires_grad=False)
    a._dtype = "float32"
    b._dtype = "float32"
    
    def _fake_mul(mask):
        return _FakeGrad(5.0)
    
    grad_output = _FakeGrad(5.0)
    grad_output.mul = _fake_mul
    
    res._node = _Node(res, lambda g: autograd_rules._grad_maximum(g, a, b), parents)
    
    _backward_from_tensor(res, gradient=grad_output)

    assert a.grad is None
    assert b.grad is not None
    assert b.grad.value == 5.0


def test_grad_minimum_alignment():
    a = _FakeTensor([1], requires_grad=False)
    b = _FakeTensor([1], requires_grad=True)

    parents = [a, b]
    res = _FakeTensor([1], requires_grad=True)

    a.le = lambda other: _FakeTensor([1], requires_grad=False)
    b.lt = lambda other: _FakeTensor([1], requires_grad=False)
    a._dtype = "float32"
    b._dtype = "float32"

    def _fake_mul(mask: _FakeTensor) -> _FakeGrad:
        return _FakeGrad(7.0)

    grad_output = _FakeGrad(7.0)
    grad_output.mul = _fake_mul

    res._node = _Node(res, lambda g: autograd_rules._grad_minimum(g, a, b), parents)

    _backward_from_tensor(res, gradient=grad_output)

    assert a.grad is None
    assert b.grad is not None
    assert b.grad.value == 7.0


def test_grad_cat_alignment():
    a = _FakeTensor([2], requires_grad=False)
    b = _FakeTensor([3], requires_grad=True)
    
    parents = [a, b]
    res = _FakeTensor([5], requires_grad=True)
    
    def _fake_slice(dim, start, end):
        return _FakeGrad([1.0] * (end - start))
        
    grad_output = _FakeGrad([1.0] * 5)
    grad_output.slice = _fake_slice
    
    res._node = _Node(res, lambda g: autograd_rules._grad_cat(g, [a, b], 0), parents)
    
    _backward_from_tensor(res, gradient=grad_output)
    
    assert a.grad is None
    assert b.grad is not None
    assert b.grad.tolist() == [1.0, 1.0, 1.0]


def test_grad_scatter_input_and_src(monkeypatch):
    monkeypatch.setattr(torch.tensor_factories_ops, "tensor_from_data", lambda data, shape, dtype: _FakeTensor(shape, dtype=dtype, data=data))

    input_tensor = _FakeTensor([3], requires_grad=True)
    dim = 0
    index = _FakeTensor([2], requires_grad=False)
    index.tolist = lambda: [0, 2]
    
    src = _FakeTensor([2], requires_grad=True)
    src.tolist = lambda: [10.0, 20.0]
    
    parents = [input_tensor, src]
    res = _FakeTensor([3], requires_grad=True)
    
    grad_output = _FakeGrad([1.0, 1.0, 1.0])
    
    res._node = _Node(res, lambda g: autograd_rules._grad_scatter(g, input_tensor, dim, index, src), parents)
    
    _backward_from_tensor(res, gradient=grad_output)
    
    assert input_tensor.grad is not None
    assert input_tensor.grad.tolist() == [0.0, 1.0, 0.0]
    
    assert src.grad is not None
    assert src.grad.tolist() == [1.0, 1.0]


def test_expand_shape_normalization(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.tensor_ops, "expand_from_tensor", lambda tensor, shape: calls.append(shape))
    
    from torch._tensor import Tensor
    t = Tensor.__new__(Tensor)
    t._id = 1
    t._shape = [3]
    t._dtype = "float32"
    t._requires_grad = False
    
    t.expand([3, 3])
    t.expand(3, 3)
    
    assert calls == [[3, 3], [3, 3]]
