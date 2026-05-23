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
