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


def replication_pad_from_tensor(
    tensor: "Tensor",
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> "Tensor":
    """Replication (edge) padding backed by the dedicated GPU shader.

    Pads the last and second-to-last spatial dimensions with the replication
    of their edge values. Returns a tensor whose shape has the padded spatial
    dims grown by ``left+right`` / ``top+bottom``.
    """
    meta = _run_js_awaitable(
        _get_runtime().replicationPad(tensor._id, int(left), int(right), int(top), int(bottom))
    )
    result = _mk_tensor(meta)
    # The runtime returns a 4D (N, C, H, W) tensor; reshape back to the
    # expected (possibly lower-rank) output shape.
    shape = list(tensor._shape)
    out_shape = list(shape)
    out_shape[-1] = out_shape[-1] + left + right
    if top or bottom:
        out_shape[-2] = out_shape[-2] + top + bottom
    return result.reshape(out_shape)


def _pad2d_from_tensor(
    tensor: "Tensor",
    left: int,
    right: int,
    top: int,
    bottom: int,
    runtime_method: str,
    value: float = 0.0,
) -> "Tensor":
    """Shared bridge for the reflection / circular / constant GPU pad shaders.

    Pads the last and second-to-last spatial dimensions; returns a tensor whose
    shape has the padded spatial dims grown by ``left+right`` / ``top+bottom``.
    """
    meta = _run_js_awaitable(
        getattr(_get_runtime(), runtime_method)(
            tensor._id, int(left), int(right), int(top), int(bottom), float(value)
        )
    )
    result = _mk_tensor(meta)
    shape = list(tensor._shape)
    out_shape = list(shape)
    out_shape[-1] = out_shape[-1] + left + right
    if top or bottom:
        out_shape[-2] = out_shape[-2] + top + bottom
    return result.reshape(out_shape)


def reflection_pad_from_tensor(
    tensor: "Tensor", left: int, right: int, top: int, bottom: int
) -> "Tensor":
    return _pad2d_from_tensor(tensor, left, right, top, bottom, "reflectionPad")


def circular_pad_from_tensor(
    tensor: "Tensor", left: int, right: int, top: int, bottom: int
) -> "Tensor":
    return _pad2d_from_tensor(tensor, left, right, top, bottom, "circularPad")


def constant_pad_from_tensor(
    tensor: "Tensor", left: int, right: int, top: int, bottom: int, value: float = 0.0
) -> "Tensor":
    return _pad2d_from_tensor(tensor, left, right, top, bottom, "constantPad", value)


def upsample2d_from_tensor(
    tensor: "Tensor",
    out_h: int,
    out_w: int,
    mode: str,
    align_corners: bool,
) -> "Tensor":
    """2D upsample (nearest / bilinear) backed by the dedicated GPU shader.

    `mode` is ``"nearest"`` or ``"bilinear"``; ``align_corners`` matches the
    PyTorch coordinate mapping.
    """
    mode_id = 1 if mode == "bilinear" else 0
    meta = _run_js_awaitable(
        _get_runtime().upsample2d(tensor._id, int(out_h), int(out_w), mode_id, 1 if align_corners else 0)
    )
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
    from .autograd import _Node, is_grad_enabled, _grad_clamp

    meta = _run_js_awaitable(_get_runtime().clamp(tensor._id, float(min_), float(max_)))
    result = _mk_tensor(meta)
    if is_grad_enabled() and tensor._requires_grad:
        result._requires_grad = True
        result._node = _Node(result, lambda g: (_grad_clamp(g, tensor, float(min_), float(max_)),), [tensor])
    return result


def argmax_from_tensor(tensor: "Tensor", dim: int | None = None, keepdim: bool = False) -> "Tensor":
    if dim is not None:
        from .tensor_ops import topk_from_tensor

        d = dim if dim >= 0 else dim + len(tensor._shape)
        if d < 0 or d >= len(tensor._shape):
            raise IndexError(f"Dimension out of range (expected to be in range of [{-len(tensor._shape)}, {len(tensor._shape) - 1}], got {dim})")
        _, indices = topk_from_tensor(tensor, 1, dim=d, largest=True)
        return indices if keepdim else indices.squeeze(d)
    meta = _run_js_awaitable(_get_runtime().argmax(tensor._id))
    return _mk_tensor(meta)


def argmin_from_tensor(tensor: "Tensor", dim: int | None = None, keepdim: bool = False) -> "Tensor":
    if dim is not None:
        from .tensor_ops import topk_from_tensor

        d = dim if dim >= 0 else dim + len(tensor._shape)
        if d < 0 or d >= len(tensor._shape):
            raise IndexError(f"Dimension out of range (expected to be in range of [{-len(tensor._shape)}, {len(tensor._shape) - 1}], got {dim})")
        _, indices = topk_from_tensor(tensor, 1, dim=d, largest=False)
        return indices if keepdim else indices.squeeze(d)
    meta = _run_js_awaitable(_get_runtime().argmin(tensor._id))
    return _mk_tensor(meta)


def t_from_tensor(tensor: "Tensor") -> "Tensor":
    from .autograd import _Node, is_grad_enabled, _grad_transpose

    meta = _run_js_awaitable(_get_runtime().transpose2d(tensor._id))
    result = _mk_tensor(meta)
    if is_grad_enabled() and tensor._requires_grad:
        result._requires_grad = True
        result._node = _Node(result, lambda g: (_grad_transpose(g, tensor, 0, 1),), [tensor])
    return result


def tolist_from_tensor(tensor: "Tensor") -> object:
    result = _run_js_awaitable(_get_runtime().toList(tensor._id))
    flat = list(result.to_py() if hasattr(result, "to_py") else result)
    return _reshape_flat_values(flat, tensor._shape, tensor._dtype)


def destroy_tensor(tensor: "Tensor") -> None:
    _run_js_awaitable(_get_runtime().destroy(tensor._id))


def repeat_from_tensor(tensor: "Tensor", sizes: list[int]) -> "Tensor":
    from .autograd import _Node, is_grad_enabled, _grad_repeat

    meta = _run_js_awaitable(_get_runtime().repeat(tensor._id, [int(s) for s in sizes]))
    result = _mk_tensor(meta)
    if is_grad_enabled() and tensor._requires_grad:
        result._requires_grad = True
        result._node = _Node(result, lambda g: (_grad_repeat(g, tensor, sizes),), [tensor])
    return result


def reshape_from_tensor(tensor: "Tensor", shape: int | list[int] | tuple[int, ...]) -> "Tensor":
    from .autograd import _Node, is_grad_enabled, _grad_reshape

    normalized = _normalize_shape(shape)
    meta = _run_js_awaitable(_get_runtime().reshape(tensor._id, normalized))
    result = _mk_tensor(meta)
    if is_grad_enabled() and tensor._requires_grad:
        result._requires_grad = True
        result._node = _Node(result, lambda g: (_grad_reshape(g, tensor, normalized),), [tensor])
    return result


def flatten_from_tensor(tensor: "Tensor", start_dim: int = 0, end_dim: int = -1) -> "Tensor":
    from .autograd import _Node, is_grad_enabled, _grad_reshape

    meta = _run_js_awaitable(_get_runtime().flatten(tensor._id, int(start_dim), int(end_dim)))
    result = _mk_tensor(meta)
    if is_grad_enabled() and tensor._requires_grad:
        result._requires_grad = True
        result._node = _Node(result, lambda g: (_grad_reshape(g, tensor, result.shape),), [tensor])
    return result


def squeeze_from_tensor(tensor: "Tensor", dim: int | None = None) -> "Tensor":
    from .autograd import _Node, is_grad_enabled, _grad_squeeze

    if dim is None:
        meta = _run_js_awaitable(_get_runtime().squeeze(tensor._id))
    else:
        meta = _run_js_awaitable(_get_runtime().squeeze(tensor._id, int(dim)))
    result = _mk_tensor(meta)
    if is_grad_enabled() and tensor._requires_grad:
        result._requires_grad = True
        result._node = _Node(result, lambda g: (_grad_squeeze(g, tensor, dim),), [tensor])
    return result


def unsqueeze_from_tensor(tensor: "Tensor", dim: int) -> "Tensor":
    from .autograd import _Node, is_grad_enabled, _grad_unsqueeze

    meta = _run_js_awaitable(_get_runtime().unsqueeze(tensor._id, int(dim)))
    result = _mk_tensor(meta)
    if is_grad_enabled() and tensor._requires_grad:
        result._requires_grad = True
        result._node = _Node(result, lambda g: (_grad_unsqueeze(g, tensor, dim),), [tensor])
    return result


def transpose_from_tensor(tensor: "Tensor", dim0: int, dim1: int) -> "Tensor":
    from .autograd import _Node, is_grad_enabled, _grad_transpose

    meta = _run_js_awaitable(_get_runtime().transpose(tensor._id, int(dim0), int(dim1)))
    result = _mk_tensor(meta)
    if is_grad_enabled() and tensor._requires_grad:
        result._requires_grad = True
        result._node = _Node(result, lambda g: (_grad_transpose(g, tensor, dim0, dim1),), [tensor])
    return result


def permute_from_tensor(tensor: "Tensor", dims: list[int] | tuple[int, ...]) -> "Tensor":
    from .autograd import _Node, is_grad_enabled, _grad_permute

    normalized = [int(v) for v in dims]
    meta = _run_js_awaitable(_get_runtime().permute(tensor._id, normalized))
    result = _mk_tensor(meta)
    if is_grad_enabled() and tensor._requires_grad:
        result._requires_grad = True
        result._node = _Node(result, lambda g: (_grad_permute(g, tensor, normalized),), [tensor])
    return result
