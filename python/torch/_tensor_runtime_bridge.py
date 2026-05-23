from __future__ import annotations

from typing import TYPE_CHECKING

from ._runtime import _get_runtime, _run_js_awaitable
from .tensor_ops import _js_meta_to_tuple
from .tensor_shape_utils import _normalize_shape, _reshape_flat_values

if TYPE_CHECKING:
    from ._tensor import Tensor


def _mk_tensor(meta: object) -> "Tensor":
    from ._tensor import Tensor

    tensor_id, shape, dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, shape, dtype)


def cholesky_from_tensor(tensor: "Tensor") -> "Tensor":
    meta = _run_js_awaitable(_get_runtime().cholesky(tensor._id))
    return _mk_tensor(meta)


def lu_from_tensor(tensor: "Tensor") -> tuple["Tensor", "Tensor"]:
    result = _run_js_awaitable(_get_runtime().lu(tensor._id))
    return _mk_tensor(result[0]), _mk_tensor(result[1])


def triangular_solve_from_tensors(a: "Tensor", b: "Tensor", upper: bool = False) -> "Tensor":
    meta = _run_js_awaitable(_get_runtime().triangularSolve(a._id, b._id, upper))
    return _mk_tensor(meta)


def item_from_tensor(tensor: "Tensor") -> float:
    return _run_js_awaitable(_get_runtime().toList(tensor._id))[0]


def clamp_from_tensor(tensor: "Tensor", min_: float, max_: float) -> "Tensor":
    meta = _run_js_awaitable(_get_runtime().clamp(tensor._id, float(min_), float(max_)))
    return _mk_tensor(meta)


def argmax_from_tensor(tensor: "Tensor") -> "Tensor":
    meta = _run_js_awaitable(_get_runtime().argmax(tensor._id))
    return _mk_tensor(meta)


def argmin_from_tensor(tensor: "Tensor") -> "Tensor":
    meta = _run_js_awaitable(_get_runtime().argmin(tensor._id))
    return _mk_tensor(meta)


def t_from_tensor(tensor: "Tensor") -> "Tensor":
    meta = _run_js_awaitable(_get_runtime().transpose2d(tensor._id))
    return _mk_tensor(meta)


def tolist_from_tensor(tensor: "Tensor") -> object:
    result = _run_js_awaitable(_get_runtime().toList(tensor._id))
    flat = list(result.to_py() if hasattr(result, "to_py") else result)
    return _reshape_flat_values(flat, tensor._shape, tensor._dtype)


def destroy_tensor(tensor: "Tensor") -> None:
    _run_js_awaitable(_get_runtime().destroy(tensor._id))


def repeat_from_tensor(tensor: "Tensor", sizes: list[int]) -> "Tensor":
    meta = _run_js_awaitable(_get_runtime().repeat(tensor._id, [int(s) for s in sizes]))
    return _mk_tensor(meta)


def reshape_from_tensor(tensor: "Tensor", shape: int | list[int] | tuple[int, ...]) -> "Tensor":
    normalized = _normalize_shape(shape)
    meta = _run_js_awaitable(_get_runtime().reshape(tensor._id, normalized))
    return _mk_tensor(meta)


def flatten_from_tensor(tensor: "Tensor", start_dim: int = 0, end_dim: int = -1) -> "Tensor":
    meta = _run_js_awaitable(_get_runtime().flatten(tensor._id, int(start_dim), int(end_dim)))
    return _mk_tensor(meta)


def squeeze_from_tensor(tensor: "Tensor", dim: int | None = None) -> "Tensor":
    if dim is None:
        meta = _run_js_awaitable(_get_runtime().squeeze(tensor._id))
    else:
        meta = _run_js_awaitable(_get_runtime().squeeze(tensor._id, int(dim)))
    return _mk_tensor(meta)


def unsqueeze_from_tensor(tensor: "Tensor", dim: int) -> "Tensor":
    meta = _run_js_awaitable(_get_runtime().unsqueeze(tensor._id, int(dim)))
    return _mk_tensor(meta)


def transpose_from_tensor(tensor: "Tensor", dim0: int, dim1: int) -> "Tensor":
    meta = _run_js_awaitable(_get_runtime().transpose(tensor._id, int(dim0), int(dim1)))
    return _mk_tensor(meta)


def permute_from_tensor(tensor: "Tensor", dims: list[int] | tuple[int, ...]) -> "Tensor":
    normalized = [int(v) for v in dims]
    meta = _run_js_awaitable(_get_runtime().permute(tensor._id, normalized))
    return _mk_tensor(meta)
