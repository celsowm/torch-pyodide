from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .tensor_shape_utils import _infer_shape, _flatten, _normalize_shape

if TYPE_CHECKING:
    from ._tensor import Tensor


def _mk(meta: object, requires_grad: bool = False) -> "Tensor":
    from ._tensor import Tensor
    from .tensor_ops import _js_meta_to_tuple
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype, _requires_grad=requires_grad)


def tensor_from_data(data: object, shape: Sequence[int] | None = None, dtype: str = "float32", requires_grad: bool = False) -> "Tensor":
    from ._runtime import _get_runtime, _run_js_awaitable
    runtime = _get_runtime()
    if shape is None:
        shape = _infer_shape(data)
    flat = _flatten(data)
    meta = _run_js_awaitable(runtime.tensorFromData(flat, shape, dtype))
    return _mk(meta, requires_grad=requires_grad)


def zeros_from_shape(shape: int | Sequence[int], dtype: str = "float32") -> "Tensor":
    from ._runtime import _get_runtime, _run_js_awaitable
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.zeros(_normalize_shape(shape), dtype))
    return _mk(meta)


def ones_from_shape(shape: int | Sequence[int], dtype: str = "float32") -> "Tensor":
    from ._runtime import _get_runtime, _run_js_awaitable
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.ones(_normalize_shape(shape), dtype))
    return _mk(meta)


def rand_from_shape(shape: int | Sequence[int], dtype: str = "float32") -> "Tensor":
    from ._runtime import _get_runtime, _run_js_awaitable
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.rand(_normalize_shape(shape), dtype))
    return _mk(meta)


def randn_from_shape(shape: int | Sequence[int], dtype: str = "float32") -> "Tensor":
    from ._runtime import _get_runtime, _run_js_awaitable
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.randn(_normalize_shape(shape), dtype))
    return _mk(meta)


def arange_from_values(start: float, end: float | None = None, step: float = 1.0, dtype: str = "float32") -> "Tensor":
    from ._runtime import _get_runtime, _run_js_awaitable
    runtime = _get_runtime()
    resolved_start, resolved_end = (0.0, float(start)) if end is None else (float(start), float(end))
    meta = _run_js_awaitable(runtime.arange(resolved_start, resolved_end, float(step), dtype))
    return _mk(meta)


def full_from_shape(shape: int | Sequence[int], fill_value: float, dtype: str = "float32") -> "Tensor":
    from ._runtime import _get_runtime, _run_js_awaitable
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.full(_normalize_shape(shape), float(fill_value), dtype))
    return _mk(meta)


def full_like_from_tensor(tensor: "Tensor", fill_value: float, dtype: str | None = None) -> "Tensor":
    from ._runtime import _get_runtime, _run_js_awaitable
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.fullLike(tensor._id, float(fill_value), dtype)) if dtype is not None else _run_js_awaitable(runtime.fullLike(tensor._id, float(fill_value)))
    return _mk(meta)


def zeros_like_from_tensor(tensor: "Tensor", dtype: str | None = None) -> "Tensor":
    from ._runtime import _get_runtime, _run_js_awaitable
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.zerosLike(tensor._id, dtype)) if dtype is not None else _run_js_awaitable(runtime.zerosLike(tensor._id))
    return _mk(meta)


def ones_like_from_tensor(tensor: "Tensor", dtype: str | None = None) -> "Tensor":
    from ._runtime import _get_runtime, _run_js_awaitable
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.onesLike(tensor._id, dtype)) if dtype is not None else _run_js_awaitable(runtime.onesLike(tensor._id))
    return _mk(meta)


def empty_like_from_tensor(tensor: "Tensor", dtype: str | None = None) -> "Tensor":
    from ._runtime import _get_runtime, _run_js_awaitable
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.emptyLike(tensor._id, dtype)) if dtype is not None else _run_js_awaitable(runtime.emptyLike(tensor._id))
    return _mk(meta)


def empty_from_shape(shape: int | Sequence[int], dtype: str = "float32") -> "Tensor":
    from ._runtime import _get_runtime, _run_js_awaitable
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.empty(_normalize_shape(shape), dtype))
    return _mk(meta)


def normal_from_shape(shape: int | Sequence[int], mean: float, std: float, dtype: str = "float32") -> "Tensor":
    from ._runtime import _get_runtime, _run_js_awaitable
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.normal(_normalize_shape(shape), dtype, float(mean), float(std)))
    return _mk(meta)


def bernoulli_from_shape(shape: int | Sequence[int], p: float = 0.5, dtype: str = "float32") -> "Tensor":
    from ._runtime import _get_runtime, _run_js_awaitable
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.bernoulli(_normalize_shape(shape), dtype, float(p)))
    return _mk(meta)


def exponential_from_shape(shape: int | Sequence[int], lambd: float = 1.0, dtype: str = "float32") -> "Tensor":
    from ._runtime import _get_runtime, _run_js_awaitable
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.exponential(_normalize_shape(shape), dtype, float(lambd)))
    return _mk(meta)


def log_normal_from_shape(shape: int | Sequence[int], mean: float = 0.0, std: float = 1.0, dtype: str = "float32") -> "Tensor":
    from ._runtime import _get_runtime, _run_js_awaitable
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.logNormal(_normalize_shape(shape), dtype, float(mean), float(std)))
    return _mk(meta)
