import pytest

import torch
import torch._api_creation as creation
import torch.tensor_factories_ops as factory_ops
import torch.tensor_ops as tensor_ops
from torch.tensor_shape_utils import _infer_shape
from torch._tensor import Tensor


@pytest.fixture(autouse=True)
def fake_creation_factories(monkeypatch):
    def make_factory(name):
        def factory(shape, dtype="float32"):
            return Tensor(hash(name) & 0xFFFF, list(shape), dtype)

        return factory

    def tensor_factory(data, shape=None, dtype="float32", requires_grad=False):
        return Tensor(hash("tensor") & 0xFFFF, list(shape or _infer_shape(data)), dtype, _requires_grad=requires_grad)

    monkeypatch.setattr(creation, "empty_from_shape", make_factory("empty"))
    monkeypatch.setattr(creation, "ones_from_shape", make_factory("ones"))
    monkeypatch.setattr(creation, "rand_from_shape", make_factory("rand"))
    monkeypatch.setattr(creation, "randn_from_shape", make_factory("randn"))
    monkeypatch.setattr(creation, "tensor_from_data", tensor_factory)
    monkeypatch.setattr(creation, "zeros_from_shape", make_factory("zeros"))

    monkeypatch.setattr(factory_ops, "zeros_like_from_tensor", lambda tensor, dtype=None: Tensor(tensor._id, list(tensor.shape), dtype or tensor.dtype))
    monkeypatch.setattr(tensor_ops, "add_from_tensors", lambda left, right: left)


@pytest.mark.parametrize(
    ("factory", "args", "expected_shape"),
    [
        (torch.randn, (2, 3, 4), (2, 3, 4)),
        (torch.rand, (2, 3), (2, 3)),
        (torch.zeros, (2, 3), (2, 3)),
        (torch.ones, (2, 3), (2, 3)),
        (torch.empty, (2, 3), (2, 3)),
        (torch.randn, ((2, 3),), (2, 3)),
        (torch.randn, ([2, 3],), (2, 3)),
    ],
)
def test_shape_factories_accept_varargs_and_sequence_shapes(factory, args, expected_shape):
    assert factory(*args).shape == expected_shape


def test_shape_factories_reject_negative_vararg_dimensions():
    with pytest.raises(ValueError, match=">= 0"):
        torch.randn(2, -3)


def test_shape_factories_support_requires_grad_with_varargs():
    result = torch.randn(2, 3, requires_grad=True)

    assert result.shape == (2, 3)
    assert result.requires_grad is True


def test_shape_factories_support_keyword_dtype_with_varargs():
    result = torch.ones(2, 3, dtype="int32")

    assert result.shape == (2, 3)
    assert result.dtype == "int32"


def test_shape_factories_preserve_positional_dtype_compatibility():
    result = torch.zeros(2, 3, "int32")

    assert result.shape == (2, 3)
    assert result.dtype == "int32"


def test_tensor_accepts_cpu_device_and_exposes_device_property():
    result = torch.tensor([[10, 11], [20, 21]], dtype=torch.long, device="cpu")

    assert result.shape == (2, 2)
    assert result.dtype == "int64"
    assert result.device == "cpu"


def test_randn_accepts_cpu_device():
    result = torch.randn(2, 4, 8, dtype=torch.float32, device="cpu")

    assert result.shape == (2, 4, 8)
    assert result.dtype == "float32"
    assert result.device == "cpu"


def test_string_like_cpu_device_is_accepted():
    class CpuDevice:
        def __str__(self):
            return "cpu"

    result = torch.zeros(2, 3, device=CpuDevice())

    assert result.shape == (2, 3)
    assert result.device == "cpu"


def test_float_long_roundtrip_keeps_expected_dtype():
    idx = torch.tensor([[10, 11], [20, 21]], dtype=torch.long, device="cpu")
    idx_as_float = idx.float()
    idx_roundtrip = idx_as_float.long()

    assert idx_as_float.dtype == "float32"
    assert idx_roundtrip.dtype == "int64"
    assert idx_roundtrip.device == "cpu"


@pytest.mark.parametrize("device", ["cuda", "webgpu", 1])
def test_unsupported_creation_devices_raise_clear_error(device):
    with pytest.raises(RuntimeError, match="Only CPU device is supported"):
        torch.tensor([1, 2, 3], device=device)
