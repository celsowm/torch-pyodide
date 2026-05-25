import pytest

from torch._tensor import Tensor
from torch.tensor_ops import _js_meta_to_tuple
from torch.tensor_shape_utils import _flatten, _infer_shape, _normalize_shape, _reshape_flat_values


def test_scalar_tensor_numeric_conversions_delegate_to_item(monkeypatch):
    monkeypatch.setattr(Tensor, "item", lambda self: 3.0)

    tensor = Tensor.__new__(Tensor)

    assert int(tensor) == 3
    assert float(tensor) == 3.0
    assert tensor.__index__() == 3


def test_infer_shape_rectangular():
    assert _infer_shape([[1, 2], [3, 4]]) == [2, 2]


def test_flatten_nested():
    assert _flatten([[1, 2], [3, 4]]) == [1.0, 2.0, 3.0, 4.0]


def test_normalize_shape_sequence():
    assert _normalize_shape((2, 3)) == [2, 3]


def test_infer_shape_rejects_ragged():
    with pytest.raises(ValueError, match="rectangular"):
        _infer_shape([[1], [2, 3]])


def test_normalize_shape_rejects_negative():
    with pytest.raises(ValueError, match=">= 0"):
        _normalize_shape((-1, 2))


def test_js_meta_to_tuple_from_dict():
    assert _js_meta_to_tuple({"id": 7, "shape": [2, 2], "dtype": "float32"}) == (
        7,
        [2, 2],
        "float32",
    )


def test_js_meta_to_tuple_from_proxy_like():
    class ProxyMeta:
        id = 9
        shape = [3, 1]
        dtype = "float32"

    assert _js_meta_to_tuple(ProxyMeta()) == (9, [3, 1], "float32")


def test_reshape_flat_values_scalar():
    assert _reshape_flat_values([28.0], [], "float32") == 28.0


def test_reshape_flat_values_nested():
    assert _reshape_flat_values([1.0, 2.0, 3.0, 4.0], [2, 2], "float32") == [[1.0, 2.0], [3.0, 4.0]]
