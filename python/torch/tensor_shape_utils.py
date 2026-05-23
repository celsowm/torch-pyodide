from __future__ import annotations
from typing import Sequence

def _flatten_out(data: object) -> list[float]:
    if isinstance(data, list):
        out: list[float] = []
        for item in data:
            out.extend(_flatten_out(item))
        return out
    return [float(data)]


def _scalar_to_tensor(value: float, dtype: str = "float32") -> Tensor:
    from ._runtime import _get_runtime, _run_js_awaitable
    from ._tensor import Tensor
    from .tensor_ops import _js_meta_to_tuple
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.zeros([1], dtype))
    t_id, _, _ = _js_meta_to_tuple(meta)
    # Put the value in
    meta2 = _run_js_awaitable(runtime.fill(t_id, float(value)))
    t_id2, _, _ = _js_meta_to_tuple(meta2)
    return Tensor(t_id2, [1], dtype)


def _infer_shape(data: object) -> list[int]:
    if not isinstance(data, list):
        return []
    if len(data) == 0:
        return [0]
    first_shape = _infer_shape(data[0])
    for item in data[1:]:
        if _infer_shape(item) != first_shape:
            raise ValueError("tensor() expects a rectangular nested list.")
    return [len(data), *first_shape]


def _flatten(data: object) -> list[float]:
    if isinstance(data, list):
        out: list[float] = []
        for item in data:
            out.extend(_flatten(item))
        return out
    return [float(data)]  # type: ignore[arg-type]


def _normalize_shape(shape: int | Sequence[int]) -> list[int]:
    if isinstance(shape, int):
        if shape < 0:
            raise ValueError("shape values must be >= 0")
        return [shape]

    normalized = [int(v) for v in shape]
    if any(v < 0 for v in normalized):
        raise ValueError("shape values must be >= 0")
    return normalized


def _normalize_shape_from_args(shape: Sequence[int]) -> list[int]:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        return _normalize_shape(shape[0])
    return _normalize_shape([int(v) for v in shape])


def _coerce_out_value(value: float, dtype: str) -> object:
    if dtype == "bool":
        return bool(value)
    if dtype == "int32":
        return int(value)
    return float(value)


def _reshape_flat_values(flat: list[float], shape: Sequence[int], dtype: str = "float32") -> object:
    if len(shape) == 0:
        return _coerce_out_value(float(flat[0]), dtype) if flat else _coerce_out_value(0.0, dtype)
    if len(shape) == 1:
        width = int(shape[0])
        return [_coerce_out_value(float(v), dtype) for v in flat[:width]]
    stride = 1
    for dim in shape[1:]:
        stride *= int(dim)
    width = int(shape[0])
    out: list[object] = []
    for i in range(width):
        start = i * stride
        out.append(_reshape_flat_values(flat[start : start + stride], shape[1:], dtype))
    return out


