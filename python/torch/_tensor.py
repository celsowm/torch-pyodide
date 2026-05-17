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

    def view(self, *shape: int) -> "Tensor":
        normalized = _normalize_shape_from_args(shape)
        return self.reshape(normalized)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.flatten(self._id, int(start_dim), int(end_dim)))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def squeeze(self, dim: int | None = None) -> "Tensor":
        runtime = _get_runtime()
        if dim is None:
            meta = _run_js_awaitable(runtime.squeeze(self._id))
        else:
            meta = _run_js_awaitable(runtime.squeeze(self._id, int(dim)))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def unsqueeze(self, dim: int) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.unsqueeze(self._id, int(dim)))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.transpose(self._id, int(dim0), int(dim1)))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def permute(self, dims: Sequence[int]) -> "Tensor":
        runtime = _get_runtime()
        normalized = [int(v) for v in dims]
        meta = _run_js_awaitable(runtime.permute(self._id, normalized))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def expand(self, *shape: int) -> "Tensor":
        runtime = _get_runtime()
        normalized = _normalize_shape_from_args(shape)
        meta = _run_js_awaitable(runtime.expand(self._id, normalized))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def select(self, dim: int, index: int) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.select(self._id, int(dim), int(index)))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def slice(self, dim: int, start: int | None = None, end: int | None = None, step: int = 1) -> "Tensor":
        runtime = _get_runtime()
        if start is None and end is None:
            meta = _run_js_awaitable(runtime.slice(self._id, int(dim), None, None, int(step)))
        elif end is None:
            meta = _run_js_awaitable(runtime.slice(self._id, int(dim), int(start), None, int(step)))
        else:
            meta = _run_js_awaitable(runtime.slice(self._id, int(dim), int(start) if start is not None else None, int(end), int(step)))
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

    def destroy(self) -> None:
        runtime = _get_runtime()
        _run_js_awaitable(runtime.destroy(self._id))

    def sigmoid(self) -> "Tensor":
        return sigmoid_from_tensor(self)

    def tanh(self) -> "Tensor":
        return tanh_from_tensor(self)

    def sin(self) -> "Tensor":
        return sin_from_tensor(self)

    def cos(self) -> "Tensor":
        return cos_from_tensor(self)

    def gelu(self) -> "Tensor":
        return gelu_from_tensor(self)

    def silu(self) -> "Tensor":
        return silu_from_tensor(self)

    def leaky_relu(self, alpha: float = 0.01) -> "Tensor":
        return leaky_relu_from_tensor(self, alpha)

    def floor(self) -> "Tensor":
        return floor_from_tensor(self)

    def ceil(self) -> "Tensor":
        return ceil_from_tensor(self)

    def round(self) -> "Tensor":
        return round_from_tensor(self)

    def reciprocal(self) -> "Tensor":
        return reciprocal_from_tensor(self)

    def square(self) -> "Tensor":
        return square_from_tensor(self)

    def eq(self, other: "Tensor") -> "Tensor":
        return eq_from_tensors(self, other)

    def ne(self, other: "Tensor") -> "Tensor":
        return ne_from_tensors(self, other)

    def lt(self, other: "Tensor") -> "Tensor":
        return lt_from_tensors(self, other)

    def le(self, other: "Tensor") -> "Tensor":
        return le_from_tensors(self, other)

    def gt(self, other: "Tensor") -> "Tensor":
        return gt_from_tensors(self, other)

    def ge(self, other: "Tensor") -> "Tensor":
        return ge_from_tensors(self, other)

    def sum(self, dim: int | None = None, keepdim: bool = False) -> "Tensor":
        if dim is not None:
            return sum_dim_from_tensor(self, dim, keepdim)
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.sum(self._id))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def mean(self, dim: int | None = None, keepdim: bool = False) -> "Tensor":
        if dim is not None:
            return mean_dim_from_tensor(self, dim, keepdim)
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.mean(self._id))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def prod(self) -> "Tensor":
        return prod_from_tensor(self)

    def min(self) -> "Tensor":
        return min_from_tensor(self)

    def max(self) -> "Tensor":
        return max_from_tensor(self)

    def masked_select(self, mask: "Tensor") -> "Tensor":
        return masked_select_from_tensor(self, mask)

    def masked_fill(self, mask: "Tensor", value: float) -> "Tensor":
        return masked_fill_from_tensor(self, mask, value)

    def __getitem__(self, key: object) -> object:
        if isinstance(key, int):
            return self.select(0, key)
        if isinstance(key, slice):
            return self.slice(0, key.start, key.stop, 1 if key.step is None else int(key.step))
        raise TypeError("Tensor indexing supports only int or slice in MVP.")


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


def cat_from_tensors(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    runtime = _get_runtime()
    ids = [t._id for t in tensors]
    meta = _run_js_awaitable(runtime.cat(ids, int(dim)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def stack_from_tensors(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    runtime = _get_runtime()
    ids = [t._id for t in tensors]
    meta = _run_js_awaitable(runtime.stack(ids, int(dim)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def expand_from_tensor(tensor: Tensor, shape: int | Sequence[int]) -> Tensor:
    runtime = _get_runtime()
    normalized = _normalize_shape(shape)
    meta = _run_js_awaitable(runtime.expand(tensor._id, normalized))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def index_select_from_tensor(input: Tensor, dim: int, index: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.indexSelect(input._id, int(dim), index._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def sigmoid_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.sigmoid(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def tanh_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.tanh(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def sin_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.sin(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def cos_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.cos(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def gelu_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.gelu(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def silu_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.silu(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def leaky_relu_from_tensor(tensor: Tensor, alpha: float = 0.01) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.leakyRelu(tensor._id, alpha))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def floor_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.floor(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def ceil_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.ceil(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def round_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.round(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def reciprocal_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.reciprocal(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def square_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.square(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def eq_from_tensors(a: Tensor, b: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.eq(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def ne_from_tensors(a: Tensor, b: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.ne(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def lt_from_tensors(a: Tensor, b: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.lt(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def le_from_tensors(a: Tensor, b: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.le(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def gt_from_tensors(a: Tensor, b: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.gt(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def ge_from_tensors(a: Tensor, b: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.ge(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def sum_dim_from_tensor(tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.sumDim(tensor._id, int(dim), keepdim))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def mean_dim_from_tensor(tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.meanDim(tensor._id, int(dim), keepdim))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def prod_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.prod(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def min_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.min(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def max_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.max(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def masked_select_from_tensor(tensor: Tensor, mask: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.maskedSelect(tensor._id, mask._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def masked_fill_from_tensor(tensor: Tensor, mask: Tensor, value: float) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.maskedFill(tensor._id, mask._id, float(value)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)
