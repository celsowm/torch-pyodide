from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ._runtime import _get_runtime, _run_js_awaitable


def _js_meta_to_tuple(meta: object) -> tuple[int, list[int], str]:
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

    def to_list(self) -> list[float]:
        runtime = _get_runtime()
        result = _run_js_awaitable(runtime.toList(self._id))
        return list(result)

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

