from __future__ import annotations

from torch import autograd_rules


class _FakeTensor:
    def __init__(self, shape: list[int], requires_grad: bool = True) -> None:
        self._shape = shape
        self._requires_grad = requires_grad


class _FakeGrad:
    def __init__(self, name: str) -> None:
        self.name = name

    def neg(self) -> "_FakeGrad":
        return _FakeGrad(f"-{self.name}")


class _FakeReducibleTensor:
    def __init__(self, shape: list[int]) -> None:
        self._shape = shape
        self.sum_calls: list[tuple[int, bool]] = []
        self.reshape_calls: list[list[int]] = []

    def sum(self, dim: int | None = None, keepdim: bool = False) -> "_FakeReducibleTensor":
        assert dim is not None
        self.sum_calls.append((dim, keepdim))
        next_shape = list(self._shape)
        if keepdim:
            next_shape[dim] = 1
        else:
            next_shape.pop(dim)
        result = _FakeReducibleTensor(next_shape)
        result.sum_calls = self.sum_calls
        result.reshape_calls = self.reshape_calls
        return result

    def reshape(self, shape: list[int]) -> "_FakeReducibleTensor":
        self.reshape_calls.append(list(shape))
        result = _FakeReducibleTensor(list(shape))
        result.sum_calls = self.sum_calls
        result.reshape_calls = self.reshape_calls
        return result


def test_grad_add_reduces_broadcasted_shapes(monkeypatch):
    grad_output = _FakeGrad("g")
    a = _FakeTensor([4, 1], requires_grad=True)
    b = _FakeTensor([1], requires_grad=True)
    calls: list[tuple[str, ...]] = []

    def _fake_reduce(grad, target_shape):
        calls.append(("reduce", grad.name, str(target_shape)))
        return _FakeGrad(f"{grad.name}->{target_shape}")

    monkeypatch.setattr(autograd_rules, "_reduce_broadcast", _fake_reduce)

    grad_a, grad_b = autograd_rules._grad_add(grad_output, a, b)

    assert grad_a.name == "g->[4, 1]"
    assert grad_b.name == "g->[1]"
    assert calls == [
        ("reduce", "g", "[4, 1]"),
        ("reduce", "g", "[1]"),
    ]


def test_grad_sub_reduces_broadcasted_shape_before_negation(monkeypatch):
    grad_output = _FakeGrad("g")
    a = _FakeTensor([4, 1], requires_grad=True)
    b = _FakeTensor([1], requires_grad=True)

    def _fake_reduce(grad, target_shape):
        return _FakeGrad(f"{grad.name}->{target_shape}")

    monkeypatch.setattr(autograd_rules, "_reduce_broadcast", _fake_reduce)

    grad_a, grad_b = autograd_rules._grad_sub(grad_output, a, b)

    assert grad_a.name == "g->[4, 1]"
    assert grad_b.name == "-g->[1]"


def test_reduce_broadcast_collapses_expand_prefix_dimension():
    grad_output = _FakeReducibleTensor([3, 3])

    result = autograd_rules._reduce_broadcast(grad_output, [3])

    assert result._shape == [3]
    assert grad_output.sum_calls == [(0, False)]
    assert grad_output.reshape_calls == []


def test_reduce_broadcast_collapses_scalar_gradient_to_scalar_shape():
    grad_output = _FakeReducibleTensor([1])

    result = autograd_rules._reduce_broadcast(grad_output, [])

    assert result._shape == []
    assert grad_output.sum_calls == [(0, False)]
    assert grad_output.reshape_calls == []
