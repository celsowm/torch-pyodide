import pytest

import torch
import torch._api_creation as creation
from torch._tensor import Tensor


@pytest.fixture(autouse=True)
def fake_creation_factories(monkeypatch):
    def make_factory(name):
        def factory(shape, dtype="float32"):
            return Tensor(hash(name) & 0xFFFF, list(shape), dtype)

        return factory

    monkeypatch.setattr(creation, "empty_from_shape", make_factory("empty"))
    monkeypatch.setattr(creation, "ones_from_shape", make_factory("ones"))
    monkeypatch.setattr(creation, "rand_from_shape", make_factory("rand"))
    monkeypatch.setattr(creation, "randn_from_shape", make_factory("randn"))
    monkeypatch.setattr(creation, "zeros_from_shape", make_factory("zeros"))


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
