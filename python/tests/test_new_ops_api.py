"""Tests for the new math/factories ops added in the compat expansion.

These tests use a fake runtime (monkeypatched) to verify that:
  * The new top-level functions and Tensor methods exist with the expected
    signature and dispatch to the expected runtime method.
  * The new factory functions (torch.normal, torch.bernoulli,
    torch.exponential, torch.log_normal) correctly thread their parameters
    to the underlying runtime.
  * The lerp decomposition (lerp = start + weight * (end - start)) and
    addcmul/addcdiv (mul + add) compose the right runtime calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch._api_creation as creation
import torch.tensor_factories_ops as factory_ops
import torch.tensor_ops as tensor_ops
from torch._tensor import Tensor
from torch.tensor_shape_utils import _infer_shape


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self._id_counter = 1000

    def _next_handle(self) -> dict:
        self._id_counter += 1
        return {
            "id": self._id_counter,
            "shape": [2, 2],
            "dtype": "float32",
        }

    def _record(self, name: str, *args) -> dict:
        self.calls.append((name, args))
        return self._next_handle()

    # Binary elementwise ops
    def atan2(self, a, b): return self._record("atan2", a, b)
    def hypot(self, a, b): return self._record("hypot", a, b)
    def logaddexp(self, a, b): return self._record("logaddexp", a, b)
    def logaddexp2(self, a, b): return self._record("logaddexp2", a, b)
    def fmod(self, a, b): return self._record("fmod", a, b)
    def remainder(self, a, b): return self._record("remainder", a, b)
    def xlogy(self, a, b): return self._record("xlogy", a, b)
    def copysign(self, a, b): return self._record("copysign", a, b)
    def nextafter(self, a, b): return self._record("nextafter", a, b)
    def floorDivide(self, a, b): return self._record("floorDivide", a, b)
    def trueDivide(self, a, b): return self._record("trueDivide", a, b)
    def logicalAnd(self, a, b): return self._record("logicalAnd", a, b)
    def logicalOr(self, a, b): return self._record("logicalOr", a, b)
    def logicalXor(self, a, b): return self._record("logicalXor", a, b)
    def bitwiseAnd(self, a, b): return self._record("bitwiseAnd", a, b)
    def bitwiseOr(self, a, b): return self._record("bitwiseOr", a, b)
    def bitwiseXor(self, a, b): return self._record("bitwiseXor", a, b)
    def bitwiseNot(self, a): return self._record("bitwiseNot", a)

    # Comparison ops
    def eq(self, a, b): return self._record("eq", a, b)
    def ne(self, a, b): return self._record("ne", a, b)
    def lt(self, a, b): return self._record("lt", a, b)
    def le(self, a, b): return self._record("le", a, b)
    def gt(self, a, b): return self._record("gt", a, b)
    def ge(self, a, b): return self._record("ge", a, b)

    # Ternary/scalar helpers
    def add(self, a, b): return self._record("add", a, b)
    def sub(self, a, b): return self._record("sub", a, b)
    def mul(self, a, b): return self._record("mul", a, b)
    def div(self, a, b): return self._record("div", a, b)
    def maximum(self, a, b): return self._record("maximum", a, b)
    def minimum(self, a, b): return self._record("minimum", a, b)
    def mulScalar(self, a, v): return self._record("mulScalar", a, v)
    def lerpScalar(self, s, e, w):
        # Simulate the TS decomposition: sub(e, s) + mulScalar(_, w) + add(s, _)
        diff = self.sub(e, s)
        scaled = self.mulScalar(diff["id"], w)
        return self.add(s, scaled["id"])
    def lerpTensor(self, s, e, w): return self._record("lerpTensor", s, e, w)
    def addcmul(self, i, t1, t2, v):
        # Simulate the TS decomposition: mul(t1, t2) + add(i, _)
        product = self.mul(t1, t2)
        if v == 1.0:
            return self.add(i, product["id"])
        scaled = self.mulScalar(product["id"], v)
        return self.add(i, scaled["id"])
    def addcdiv(self, i, t1, t2, v):
        # Simulate the TS decomposition: div(t1, t2) + add(i, _)
        quotient = self.div(t1, t2)
        if v == 1.0:
            return self.add(i, quotient["id"])
        scaled = self.mulScalar(quotient["id"], v)
        return self.add(i, scaled["id"])

    # Misc helpers used by scalar-to-tensor conversion
    def zeros(self, shape, dtype): return self._record("zeros", shape, dtype)
    def ones(self, shape, dtype): return self._record("ones", shape, dtype)
    def fill(self, tensor_id, value): return self._record("fill", tensor_id, value)
    def tensorFromData(self, data, shape, dtype): return self._record("tensorFromData", data, shape, dtype)
    def toList(self, tensor_id):
        self.calls.append(("toList", (tensor_id,)))
        # Return a flat list of zeros matching the implied shape (2x2 for our tests).
        return [0.0, 0.0, 0.0, 0.0]

    # Factories
    def normal(self, shape, dtype, mean, std): return self._record("normal", shape, dtype, mean, std)
    def bernoulli(self, shape, dtype, p): return self._record("bernoulli", shape, dtype, p)
    def exponential(self, shape, dtype, rate): return self._record("exponential", shape, dtype, rate)
    def logNormal(self, shape, dtype, mean, std): return self._record("logNormal", shape, dtype, mean, std)
    def randn(self, shape, dtype): return self._record("randn", shape, dtype)
    def rand(self, shape, dtype): return self._record("rand", shape, dtype)


@pytest.fixture
def fake_runtime(monkeypatch):
    rt = _FakeRuntime()
    modules_with_get_runtime = [
        "torch._runtime",
        "torch._tensor",
        "torch._api_creation",
        "torch.tensor_ops",
        "torch.tensor_backward_ops",
        "torch.tensor_nn_ops",
        "torch._tensor_runtime_bridge",
        "torch._tensor_linalg_py",
    ]
    for module_path in modules_with_get_runtime:
        monkeypatch.setattr(f"{module_path}._get_runtime", lambda: rt)
    modules_with_run_awaitable = [
        "torch._runtime",
        "torch._tensor",
        "torch.tensor_ops",
        "torch.tensor_backward_ops",
        "torch.tensor_nn_ops",
        "torch._tensor_runtime_bridge",
        "torch._tensor_linalg_py",
    ]
    for module_path in modules_with_run_awaitable:
        monkeypatch.setattr(f"{module_path}._run_js_awaitable", lambda awaitable: awaitable)
    return rt


# ---------------------------------------------------------------------------
# Binary op wrappers: ensure each dispatch path is hit with the right call.
# ---------------------------------------------------------------------------

BINARY_OPS = [
    ("atan2", (1.0, 2.0)),
    ("hypot", (3.0, 4.0)),
    ("fmod", (5.0, 2.0)),
    ("remainder", (5.0, 2.0)),
    ("logaddexp", (1.0, 2.0)),
    ("logaddexp2", (1.0, 2.0)),
    ("xlogy", (2.0, 3.0)),
    ("copysign", (1.0, -1.0)),
    ("nextafter", (1.0, 2.0)),
    ("floor_divide", (5.0, 2.0)),
    ("true_divide", (5.0, 2.0)),
    ("logical_and", (1.0, 0.0)),
    ("logical_or", (0.0, 0.0)),
    ("logical_xor", (1.0, 1.0)),
    ("bitwise_and", (3, 1)),
    ("bitwise_or", (3, 1)),
    ("bitwise_xor", (3, 1)),
]


@pytest.mark.parametrize("op_name,scalar_args", BINARY_OPS)
def test_binary_op_dispatches_to_correct_runtime_method(fake_runtime, op_name, scalar_args):
    a = Tensor(1, [2], "float32")
    b = Tensor(2, [2], "float32")
    fn = getattr(torch, op_name)
    fn(a, b)
    expected = op_name
    if op_name in ("floor_divide", "true_divide"):
        expected = "floor_divide" if op_name == "floor_divide" else "trueDivide"
        # Mapping: actual runtime method names
    runtime_method_map = {
        "floor_divide": "floorDivide",
        "true_divide": "trueDivide",
        "logical_and": "logicalAnd",
        "logical_or": "logicalOr",
        "logical_xor": "logicalXor",
        "bitwise_and": "bitwiseAnd",
        "bitwise_or": "bitwiseOr",
        "bitwise_xor": "bitwiseXor",
        "logaddexp2": "logaddexp2",
    }
    expected_rt = runtime_method_map.get(op_name, op_name)
    assert any(c[0] == expected_rt for c in fake_runtime.calls), (
        f"expected runtime.{expected_rt}() to be called, got: {[c[0] for c in fake_runtime.calls]}"
    )


def test_bitwise_not_dispatches(fake_runtime):
    a = Tensor(1, [2], "float32")
    torch.bitwise_not(a)
    assert any(c[0] == "bitwiseNot" for c in fake_runtime.calls)


def test_fmax_dispatches_to_maximum(fake_runtime):
    a = Tensor(1, [2], "float32")
    b = Tensor(2, [2], "float32")
    torch.fmax(a, b)
    assert any(c[0] == "maximum" for c in fake_runtime.calls)


def test_fmin_dispatches_to_minimum(fake_runtime):
    a = Tensor(1, [2], "float32")
    b = Tensor(2, [2], "float32")
    torch.fmin(a, b)
    assert any(c[0] == "minimum" for c in fake_runtime.calls)


# ---------------------------------------------------------------------------
# lerp: scalar vs tensor path
# ---------------------------------------------------------------------------

def test_lerp_scalar_uses_sub_mul_add(fake_runtime):
    a = Tensor(1, [2], "float32")
    b = Tensor(2, [2], "float32")
    torch.lerp(a, b, 0.5)
    method_names = [c[0] for c in fake_runtime.calls]
    # Expected: sub, mulScalar, add (in that order)
    assert "sub" in method_names
    assert "mulScalar" in method_names
    assert "add" in method_names


def test_lerp_tensor_uses_ternary_shader(fake_runtime):
    a = Tensor(1, [2], "float32")
    b = Tensor(2, [2], "float32")
    w = Tensor(3, [2], "float32")
    torch.lerp(a, b, w)
    assert any(c[0] == "lerpTensor" for c in fake_runtime.calls)


# ---------------------------------------------------------------------------
# addcmul / addcdiv: compose mul + add (and mulScalar if value != 1)
# ---------------------------------------------------------------------------

def test_addcmul_value_one_uses_mul_add(fake_runtime):
    a = Tensor(1, [2], "float32")
    t1 = Tensor(2, [2], "float32")
    t2 = Tensor(3, [2], "float32")
    torch.addcmul(a, t1, t2, value=1.0)
    method_names = [c[0] for c in fake_runtime.calls]
    assert "mul" in method_names
    assert "add" in method_names
    # value=1.0 -> no mulScalar
    assert "mulScalar" not in method_names


def test_addcmul_value_nonzero_uses_mul_scalar_add(fake_runtime):
    a = Tensor(1, [2], "float32")
    t1 = Tensor(2, [2], "float32")
    t2 = Tensor(3, [2], "float32")
    torch.addcmul(a, t1, t2, value=2.0)
    method_names = [c[0] for c in fake_runtime.calls]
    assert "mul" in method_names
    assert "mulScalar" in method_names
    assert "add" in method_names


def test_addcdiv_value_one_uses_div_add(fake_runtime):
    a = Tensor(1, [2], "float32")
    t1 = Tensor(2, [2], "float32")
    t2 = Tensor(3, [2], "float32")
    torch.addcdiv(a, t1, t2, value=1.0)
    method_names = [c[0] for c in fake_runtime.calls]
    assert "div" in method_names
    assert "add" in method_names


# ---------------------------------------------------------------------------
# Factories: normal, bernoulli, exponential, log_normal
# ---------------------------------------------------------------------------

def test_normal_factory_dispatches_with_mean_std(fake_runtime):
    torch.normal(0.0, 1.0, size=[4, 4])
    calls = [c for c in fake_runtime.calls if c[0] == "normal"]
    assert len(calls) == 1
    args = calls[0][1]
    assert args[0] == [4, 4]
    assert args[1] == "float32"
    assert args[2] == 0.0
    assert args[3] == 1.0


def test_bernoulli_factory_dispatches_with_p(fake_runtime):
    torch.bernoulli(0.3, size=[3, 3])
    calls = [c for c in fake_runtime.calls if c[0] == "bernoulli"]
    assert len(calls) == 1
    args = calls[0][1]
    assert args[2] == 0.3


def test_exponential_factory_dispatches_with_rate(fake_runtime):
    torch.exponential(2.0, size=[2, 3])
    calls = [c for c in fake_runtime.calls if c[0] == "exponential"]
    assert len(calls) == 1
    args = calls[0][1]
    assert args[2] == 2.0


def test_log_normal_factory_dispatches_with_mean_std(fake_runtime):
    torch.log_normal(0.0, 1.0, size=[5, 5])
    calls = [c for c in fake_runtime.calls if c[0] == "logNormal"]
    assert len(calls) == 1
    args = calls[0][1]
    assert args[2] == 0.0
    assert args[3] == 1.0


def test_normal_requires_size_for_scalars():
    with pytest.raises(TypeError, match="size is required"):
        torch.normal(0.0, 1.0)


def test_bernoulli_from_tensor_does_not_call_runtime_factory(fake_runtime):
    # When given a tensor of probabilities, should NOT hit the runtime factory.
    probs = torch.tensor([[0.1, 0.5], [0.9, 0.1]])
    result = torch.bernoulli(probs)
    assert tuple(result.shape) == (2, 2)
    assert all(c[0] != "bernoulli" for c in fake_runtime.calls), (
        "tensor-input bernoulli should not invoke runtime.bernoulli()"
    )


# ---------------------------------------------------------------------------
# Tensor method smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name", [name for name, _ in BINARY_OPS] + ["bitwise_not"])
def test_tensor_method_dispatches(fake_runtime, op_name):
    a = Tensor(1, [2], "float32")
    b = Tensor(2, [2], "float32")
    if op_name == "bitwise_not":
        a.bitwise_not()
    else:
        method = getattr(a, op_name)
        method(b)
    assert len(fake_runtime.calls) >= 1


def test_tensor_atan2_accepts_scalar(fake_runtime):
    a = Tensor(1, [2], "float32")
    a.atan2(1.0)
    # Should still hit atan2 (after scalar->tensor conversion)
    assert any(c[0] == "atan2" for c in fake_runtime.calls)
