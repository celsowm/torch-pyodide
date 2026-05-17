from torch._tensor import _flatten, _infer_shape, _normalize_shape


def test_infer_shape_rectangular():
    assert _infer_shape([[1, 2], [3, 4]]) == [2, 2]


def test_flatten_nested():
    assert _flatten([[1, 2], [3, 4]]) == [1.0, 2.0, 3.0, 4.0]


def test_normalize_shape_sequence():
    assert _normalize_shape((2, 3)) == [2, 3]

