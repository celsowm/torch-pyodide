from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ._runtime import _get_runtime, _run_js_awaitable


def _js_meta_to_tuple(meta: object) -> tuple[int, list[int], str]:
    if isinstance(meta, dict):
        tensor_id = int(meta["id"])
        shape_raw = meta["shape"]
        shape = list(shape_raw.to_py() if hasattr(shape_raw, "to_py") else shape_raw)
        dtype = str(meta["dtype"])
        return tensor_id, shape, dtype

    tensor_id = int(getattr(meta, "id"))
    shape_raw = getattr(meta, "shape")
    shape = list(shape_raw.to_py() if hasattr(shape_raw, "to_py") else shape_raw)
    dtype = str(getattr(meta, "dtype"))
    return tensor_id, shape, dtype


@dataclass(slots=True)
class Tensor:
    _id: int
    _shape: list[int]
    _dtype: str

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._shape)

    @property
    def dtype(self) -> str:
        return self._dtype

    def add(self, other: "Tensor") -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.add(self._id, other._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def mul(self, other: "Tensor") -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.mul(self._id, other._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def sub(self, other: "Tensor") -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.sub(self._id, other._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def div(self, other: "Tensor") -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.div(self._id, other._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def matmul(self, other: "Tensor") -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.matmul(self._id, other._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def sum(self) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.sum(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def mean(self) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.mean(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def relu(self) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.relu(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def abs(self) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.abs(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def sqrt(self) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.sqrt(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def exp(self) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.exp(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def log(self) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.log(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def neg(self) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.neg(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def clamp(self, min: float, max: float) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.clamp(self._id, float(min), float(max)))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def argmax(self) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.argmax(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def argmin(self) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.argmin(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def reshape(self, shape: int | Sequence[int]) -> "Tensor":
        runtime = _get_runtime()
        normalized = _normalize_shape(shape)
        meta = _run_js_awaitable(runtime.reshape(self._id, normalized))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    @property
    def T(self) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.transpose2d(self._id))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def tolist(self) -> object:
        runtime = _get_runtime()
        result = _run_js_awaitable(runtime.toList(self._id))
        flat = list(result.to_py() if hasattr(result, "to_py") else result)
        return _reshape_flat_values(flat, self._shape, self._dtype)

    def to_list(self) -> object:
        return self.tolist()

    def destroy(self) -> None:
        runtime = _get_runtime()
        _run_js_awaitable(runtime.destroy(self._id))


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


def _coerce_out_value(value: float, dtype: str) -> object:
    if dtype == "bool":
        return bool(value)
    if dtype == "int32":
        return int(value)
    return float(value)


def _reshape_flat_values(flat: list[float], shape: Sequence[int], dtype: str) -> object:
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


def tensor_from_data(data: object, dtype: str = "float32") -> Tensor:
    runtime = _get_runtime()
    shape = _infer_shape(data)
    flat = _flatten(data)
    meta = _run_js_awaitable(runtime.tensorFromData(flat, shape, dtype))
    tensor_id, shape, dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, shape, dtype)


def zeros_from_shape(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    runtime = _get_runtime()
    normalized = _normalize_shape(shape)
    meta = _run_js_awaitable(runtime.zeros(normalized, dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def ones_from_shape(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    runtime = _get_runtime()
    normalized = _normalize_shape(shape)
    meta = _run_js_awaitable(runtime.ones(normalized, dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def rand_from_shape(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    runtime = _get_runtime()
    normalized = _normalize_shape(shape)
    meta = _run_js_awaitable(runtime.rand(normalized, dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def randn_from_shape(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    runtime = _get_runtime()
    normalized = _normalize_shape(shape)
    meta = _run_js_awaitable(runtime.randn(normalized, dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def arange_from_values(start: float, end: float | None = None, step: float = 1.0, dtype: str = "float32") -> Tensor:
    runtime = _get_runtime()
    if end is None:
        resolved_start = 0.0
        resolved_end = float(start)
    else:
        resolved_start = float(start)
        resolved_end = float(end)
    meta = _run_js_awaitable(runtime.arange(resolved_start, resolved_end, float(step), dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def full_from_shape(shape: int | Sequence[int], fill_value: float, dtype: str = "float32") -> Tensor:
    runtime = _get_runtime()
    normalized = _normalize_shape(shape)
    meta = _run_js_awaitable(runtime.full(normalized, float(fill_value), dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def full_like_from_tensor(tensor: Tensor, fill_value: float, dtype: str | None = None) -> Tensor:
    runtime = _get_runtime()
    if dtype is None:
        meta = _run_js_awaitable(runtime.fullLike(tensor._id, float(fill_value)))
    else:
        meta = _run_js_awaitable(runtime.fullLike(tensor._id, float(fill_value), dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def where_from_tensors(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.where(condition._id, x._id, y._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)
