from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ._runtime import _get_runtime, _run_js_awaitable

if TYPE_CHECKING:
    from ._tensor import Tensor


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


def _js_handle_array_to_tensors(meta: object) -> list["Tensor"]:
    arr = meta if isinstance(meta, list) else []
    tensors = []
    for item in arr:
        tid, shape, dtype = _js_meta_to_tuple(item)
        from ._tensor import Tensor
        tensors.append(Tensor(tid, shape, dtype))
    return tensors


def _scalar_to_tensor(value: float, dtype: str = "float32") -> "Tensor":
    from ._tensor import Tensor
    from .tensor_shape_utils import _scalar_to_tensor as _shape_scalar_to_tensor
    return _shape_scalar_to_tensor(value, dtype)


def add_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_add

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.add(a._id, b._id))
    tensor_id, shape, dtype = _js_meta_to_tuple(meta)

    requires_grad = is_grad_enabled() and (a._requires_grad or b._requires_grad)
    if requires_grad:
        result_tensor = Tensor(tensor_id, shape, dtype, _requires_grad=True)
        node = _Node(
            tensor=result_tensor,
            grad_fn=lambda grad_out: _grad_add(grad_out, a, b),
            parents=[a, b],
        )
        result_tensor._node = node
        return result_tensor

    return Tensor(tensor_id, shape, dtype)


def sub_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_sub

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.sub(a._id, b._id))
    tensor_id, shape, dtype = _js_meta_to_tuple(meta)

    requires_grad = is_grad_enabled() and (a._requires_grad or b._requires_grad)
    if requires_grad:
        result_tensor = Tensor(tensor_id, shape, dtype, _requires_grad=True)
        node = _Node(
            tensor=result_tensor,
            grad_fn=lambda grad_out: _grad_sub(grad_out, a, b),
            parents=[a, b],
        )
        result_tensor._node = node
        return result_tensor

    return Tensor(tensor_id, shape, dtype)


def mul_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_mul

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.mul(a._id, b._id))
    tensor_id, shape, dtype = _js_meta_to_tuple(meta)

    requires_grad = is_grad_enabled() and (a._requires_grad or b._requires_grad)
    if requires_grad:
        result_tensor = Tensor(tensor_id, shape, dtype, _requires_grad=True)
        node = _Node(
            tensor=result_tensor,
            grad_fn=lambda grad_out: _grad_mul(grad_out, a, b),
            parents=[a, b],
        )
        result_tensor._node = node
        return result_tensor

    return Tensor(tensor_id, shape, dtype)


def div_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_div

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.div(a._id, b._id))
    tensor_id, shape, dtype = _js_meta_to_tuple(meta)

    requires_grad = is_grad_enabled() and (a._requires_grad or b._requires_grad)
    if requires_grad:
        result_tensor = Tensor(tensor_id, shape, dtype, _requires_grad=True)
        node = _Node(
            tensor=result_tensor,
            grad_fn=lambda grad_out: _grad_div(grad_out, a, b),
            parents=[a, b],
        )
        result_tensor._node = node
        return result_tensor

    return Tensor(tensor_id, shape, dtype)


def pow_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_pow

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.pow(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and (a._requires_grad or b._requires_grad):
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: _grad_pow(g, a, b), [a, b])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def heaviside_from_tensors(input_: "Tensor", values: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.heaviside(input_._id, values._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def maximum_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_maximum

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.maximum(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and (a._requires_grad or b._requires_grad):
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        parents = [a, b]
        result._node = _Node(result, lambda g: _grad_maximum(g, a, b), parents)
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def minimum_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_minimum

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.minimum(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and (a._requires_grad or b._requires_grad):
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        parents = [a, b]
        result._node = _Node(result, lambda g: _grad_minimum(g, a, b), parents)
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def any_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.any(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def atan2_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.atan2(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def hypot_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.hypot(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def logaddexp_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.logaddexp(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def logaddexp2_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.logaddexp2(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def fmod_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.fmod(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def remainder_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.remainder(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def xlogy_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.xlogy(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def copysign_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.copysign(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def floor_divide_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.floorDivide(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def true_divide_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.trueDivide(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def nextafter_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.nextafter(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def logical_and_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.logicalAnd(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def logical_or_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.logicalOr(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def logical_xor_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.logicalXor(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def bitwise_and_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.bitwiseAnd(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def bitwise_or_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.bitwiseOr(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def bitwise_xor_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.bitwiseXor(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def bitwise_not_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.bitwiseNot(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def lerp_from_tensors(start: "Tensor", end: "Tensor", weight: "Tensor | float") -> "Tensor":
    """torch.lerp(start, end, weight): linear interpolation."""
    from ._tensor import Tensor
    runtime = _get_runtime()
    if isinstance(weight, Tensor):
        meta = _run_js_awaitable(runtime.lerpTensor(start._id, end._id, weight._id))
    else:
        meta = _run_js_awaitable(runtime.lerpScalar(start._id, end._id, float(weight)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def addcmul_from_tensors(input_: "Tensor", t1: "Tensor", t2: "Tensor", value: float = 1.0) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.addcmul(input_._id, t1._id, t2._id, float(value)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def addcdiv_from_tensors(input_: "Tensor", t1: "Tensor", t2: "Tensor", value: float = 1.0) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.addcdiv(input_._id, t1._id, t2._id, float(value)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def mul_scalar_from_tensor(tensor: "Tensor", value: float) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.mulScalar(tensor._id, float(value)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def any_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.any(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def all_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.all(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def cumsum_from_tensor(tensor: "Tensor", dim: int = 0) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_cumsum

    runtime = _get_runtime()
    shape = list(tensor._shape)
    ndim = len(shape)
    if dim < 0:
        dim += ndim

    if ndim == 1:
        meta = _run_js_awaitable(runtime.cumsum(tensor._id))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    else:
        # runtime cumsum is flat; make it dim-aware by moving the target dim
        # last, flattening, computing the flat scan, then resetting each row
        # boundary so the scan does not cross slices.
        perm = [i for i in range(ndim) if i != dim] + [dim]
        t = tensor.permute(perm)
        outer = 1
        for s in t._shape[:-1]:
            outer *= s
        L = t._shape[-1]
        flat = t.reshape([outer * L])
        meta = _run_js_awaitable(runtime.cumsum(flat._id))
        fid, _, fdtype = _js_meta_to_tuple(meta)
        flat_cum = Tensor(fid, [outer * L], fdtype)
        flat_cum_r = flat_cum.reshape([outer, L])
        if outer > 1:
            from .tensor_factories_ops import zeros_from_shape
            last_col = flat_cum_r.select(1, L - 1)
            prev_end = slice_from_tensor(last_col, 0, 0, outer - 1).reshape([outer - 1, 1])
            offset = cat_multi_from_tensors([zeros_from_shape([1, 1], fdtype), prev_end], 0)
            result_r = flat_cum_r - offset
        else:
            result_r = flat_cum_r
        result_t = result_r.reshape(list(t._shape)).permute([perm.index(i) for i in range(ndim)])
        tensor_id, out_shape, out_dtype = result_t._id, result_t._shape, result_t._dtype

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_cumsum(g, tensor, dim),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def cumprod_from_tensor(tensor: "Tensor", dim: int = 0) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_cumprod

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.cumprod(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_cumprod(g, tensor, dim),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def tril_from_tensor(tensor: "Tensor", diagonal: int = 0) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_tril

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.tril(tensor._id, int(diagonal)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_tril(g, tensor, diagonal),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def triu_from_tensor(tensor: "Tensor", diagonal: int = 0) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_triu

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.triu(tensor._id, int(diagonal)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_triu(g, tensor, diagonal),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def flip_from_tensor(tensor: "Tensor", dims: Sequence[int]) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_flip

    runtime = _get_runtime()
    normalized = [int(v) for v in dims]
    meta = _run_js_awaitable(runtime.flip(tensor._id, normalized))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_flip(g, tensor, normalized),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def where_from_tensors(condition: "Tensor", x: "Tensor", y: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_where

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.where(condition._id, x._id, y._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and (x._requires_grad or y._requires_grad):
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: _grad_where(g, condition, x, y), [condition, x, y])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def topk_from_tensor(tensor: "Tensor", k: int, dim: int = -1, largest: bool = True) -> tuple["Tensor", "Tensor"]:
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_topk
    from .tensor_shape_utils import _flatten_out
    from .tensor_factories_ops import tensor_from_data, arange_from_values

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    d = dim if dim >= 0 else dim + len(tensor._shape)
    size = tensor._shape[d]
    if k >= size:
        idx = arange_from_values(0, size, 1)
        idx_shape = [1] * len(tensor._shape)
        idx_shape[d] = size
        idx = idx.reshape(idx_shape).expand(list(tensor._shape))
        return tensor, idx
    descending = largest
    values, indices = sort_from_tensor(tensor, d, descending)
    shape = list(values._shape)
    shape[d] = k
    values_meta = _run_js_awaitable(_get_runtime().slice(values._id, int(d), 0, int(k), 1))
    indices_meta = _run_js_awaitable(_get_runtime().slice(indices._id, int(d), 0, int(k), 1))
    values_id, values_shape, values_dtype = _js_meta_to_tuple(values_meta)
    indices_id, indices_shape, indices_dtype = _js_meta_to_tuple(indices_meta)
    values_t = Tensor(values_id, values_shape, values_dtype)
    indices_t = Tensor(indices_id, indices_shape, indices_dtype)

    if is_grad_enabled() and tensor._requires_grad:
        values_t._requires_grad = True
        saved_sort_indices = indices
        values_t._node = _Node(values_t, lambda g: (_grad_topk(g, tensor, d, k, descending, saved_sort_indices),), [tensor])
    return values_t, indices_t


def sort_from_tensor(tensor: "Tensor", dim: int = -1, descending: bool = False) -> tuple["Tensor", "Tensor"]:
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_sort

    runtime = _get_runtime()
    handles = _run_js_awaitable(runtime.sort(tensor._id, int(dim)))
    tensors = _js_handle_array_to_tensors(handles)
    values, indices = tensors[0], tensors[1]
    if descending:
        values_meta = _run_js_awaitable(runtime.flip(values._id, [int(dim)]))
        indices_meta = _run_js_awaitable(runtime.flip(indices._id, [int(dim)]))
        values_id, values_shape, values_dtype = _js_meta_to_tuple(values_meta)
        indices_id, indices_shape, indices_dtype = _js_meta_to_tuple(indices_meta)
        values = Tensor(values_id, values_shape, values_dtype)
        indices = Tensor(indices_id, indices_shape, indices_dtype)

    if is_grad_enabled() and tensor._requires_grad:
        values._requires_grad = True
        saved_indices = indices
        values._node = _Node(values, lambda g: (_grad_sort(g, tensor, dim, descending, saved_indices),), [tensor])
    return values, indices


def gather_from_tensor(tensor: "Tensor", dim: int, index: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_gather

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.gather(tensor._id, int(dim), index._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_gather(g, tensor, dim, index),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def scatter_from_tensor(tensor: "Tensor", dim: int, index: "Tensor", src: "Tensor | float") -> "Tensor":
    from ._tensor import Tensor
    from ._runtime import _get_runtime, _run_js_awaitable
    from .autograd import _Node, is_grad_enabled, _grad_scatter
    from .tensor_factories_ops import full_like_from_tensor

    runtime = _get_runtime()
    # Ensure src is a tensor with the same shape as index.
    if isinstance(src, (int, float)):
        src_tensor = full_like_from_tensor(index, float(src), dtype="float32")
    else:
        src_tensor = src
    meta = _run_js_awaitable(runtime.scatter(tensor._id, int(dim), index._id, src_tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    out = Tensor(tensor_id, list(out_shape), out_dtype)

    if is_grad_enabled() and (tensor._requires_grad or (not isinstance(src, (int, float)) and src._requires_grad)):
        out._requires_grad = True
        parents = [tensor]
        if not isinstance(src, (int, float)):
            parents.append(src)
        out._node = _Node(out, lambda g: _grad_scatter(g, tensor, dim, index, src), parents)
    return out


def scatter_add_from_tensor(
    tensor: "Tensor", dim: int, index: "Tensor", src: "Tensor"
) -> "Tensor":
    """Flat scatter_add: accumulate src values at flat index positions into output."""
    from ._tensor import Tensor
    from ._runtime import _get_runtime, _run_js_awaitable

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.scatterAdd(tensor._id, int(dim), index._id, src._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, list(out_shape), out_dtype)


def scatter_add_safe_from_tensor(
    out: "Tensor", dim: int, index: "Tensor", src: "Tensor"
) -> "Tensor":
    """Atomic-free scatter_add.

    WGSL has no atomic float add and the flat ``scatter``/``scatterAdd``
    runtime shaders run one thread per source element in parallel, so when
    several ``index`` entries map to the same output position the writes race
    (the lower source index wins, discarding the others).

    To stay correct under that race we guarantee that every thread targeting
    the same output position writes the *identical* value:

      1. sort ``(index, src)`` by ``index`` (equal indices become contiguous),
      2. assign a unique group id to each contiguous block of equal indices,
      3. cumulative-sum ``src`` within the global order,
      4. keep the running sum only at each group's last position (``carry``),
      5. scatter that unique ``carry`` into a 1-slot-per-group buffer -- a
         1-to-1 write, so it is race-free,
      6. gather the per-group total back to every source element, so all
         elements of a group now carry the same total,
      7. overwrite-scatter those totals into a zero buffer and add to ``out``.
         Duplicates now write the same value, so the race is harmless.

    Equivalent to PyTorch ``out.scatter_add_(dim, index, src)``.
    """
    from .tensor_factories_ops import (
        zeros_like_from_tensor,
        zeros_from_shape,
        arange_from_values,
    )

    flat_out = out.reshape(-1)
    flat_index = index.reshape(-1).to(dtype="int64")
    flat_src = src.reshape(-1)
    n = flat_index._shape[0]
    if n == 0:
        return out

    sorted_idx, order = sort_from_tensor(flat_index, 0, descending=False)
    sorted_val = gather_from_tensor(flat_src, 0, order)

    ar = arange_from_values(0, n, 1)
    z = zeros_from_shape([n], flat_out._dtype)
    ar_first = ar.eq(z)
    ar_last = ar.ge(z.add(n - 1))

    is_first = ne_from_tensors(sorted_idx, roll_from_tensor(sorted_idx, 1)).logical_or(ar_first)
    group_id = cumsum_from_tensor(is_first.to(flat_out._dtype), 0)  # 1..G contiguous, unique per group
    G = int(group_id.select(0, n - 1).item())

    # Per-group cumulative sum. The flat ``scatter`` shader runs in parallel and
    # races on duplicate positions, so every write below is either 1-to-1 (unique
    # index) or writes the identical value for all duplicates of a group.
    running = cumsum_from_tensor(sorted_val, 0)
    is_last = ne_from_tensors(sorted_idx, roll_from_tensor(sorted_idx, -1)).logical_or(ar_last)

    # Cumulative sum just *before* each group starts (running[i-1] at group starts).
    prev_cum = roll_from_tensor(running, 1)
    prev_cum0 = where_from_tensors(ar_first, zeros_like_from_tensor(running), prev_cum)
    group_prev = scatter_from_tensor(
        zeros_from_shape([G], flat_out._dtype),
        0,
        where_from_tensors(is_first, (group_id - 1), zeros_from_shape([n], flat_out._dtype).add(float(G))),
        prev_cum0,
    )
    prev_cum_per_elem = gather_from_tensor(group_prev, 0, (group_id - 1).to("int64"))

    # Group total = cumsum at the group's last element minus the cumsum before it.
    group_total_at_last = running - where_from_tensors(is_last, prev_cum_per_elem, zeros_like_from_tensor(running))
    carry = where_from_tensors(is_last, group_total_at_last, zeros_like_from_tensor(running))

    group_totals = scatter_from_tensor(
        zeros_from_shape([G], flat_out._dtype),
        0,
        where_from_tensors(is_last, (group_id - 1), zeros_from_shape([n], flat_out._dtype).add(float(G))),
        carry,
    )
    total_per_elem = gather_from_tensor(group_totals, 0, (group_id - 1).to("int64"))

    added = scatter_from_tensor(zeros_like_from_tensor(flat_out), 0, sorted_idx, total_per_elem)
    return (flat_out + added).reshape(list(out._shape))


def _scatter_flat_positions(input_shape: list[int], dim: int, index: "Tensor") -> "Tensor":
    """Flat output positions for a multi-dim scatter.

    The runtime ``scatter`` is a flat 1-D kernel writing ``output[pos] = src[q]``
    where ``pos = flat_positions[q]``. For a scatter along ``dim`` with a 1-D
    ``index`` (length ``input_shape[dim]``), the flat position for source element
    at multi-index ``m`` is ``ravel(m)`` with ``m[dim]`` replaced by ``index[m[dim]]``.
    """
    from ._api_creation import arange

    ndim = len(input_shape)
    d = dim if dim >= 0 else dim + ndim
    total = 1
    for s in input_shape:
        total *= s
    stride_d = 1
    for s in reversed(input_shape[d + 1:]):
        stride_d *= s
    flat_q = arange(0, total, 1, dtype="int64")
    q_d = flat_q.floor_divide(stride_d)
    # Modulo without `remainder` (the runtime remainder shader is buggy for odds);
    # coord_d = q_d - k * (q_d // k).
    coord_d = q_d - input_shape[d] * q_d.floor_divide(input_shape[d])
    substituted = index.to(dtype="int64").gather(0, coord_d)
    return flat_q - coord_d * stride_d + substituted * stride_d


def index_copy_from_tensor(input: "Tensor", dim: int, index: "Tensor", source: "Tensor") -> "Tensor":
    """out = input; out.index_copy_(dim, index, source) — overwrite scatter (race-free)."""
    from .autograd import _Node, is_grad_enabled, _grad_scatter

    out_pos = _scatter_flat_positions(input._shape, dim, index)
    out = scatter_from_tensor(input.reshape(-1), 0, out_pos, source.reshape(-1))
    out = out.reshape(list(input._shape))
    if is_grad_enabled() and (input._requires_grad or source._requires_grad):
        out._requires_grad = True
        parents = [input, source]
        out._node = _Node(out, lambda g: _grad_scatter(g, input, dim, index, source), parents)
    return out


def index_fill_from_tensor(input: "Tensor", dim: int, index: "Tensor", value: "Tensor | float") -> "Tensor":
    """out = input; out.index_fill_(dim, index, value) — overwrite scatter (race-free)."""
    from .autograd import _Node, is_grad_enabled, _grad_scatter
    from .tensor_factories_ops import full_like_from_tensor

    source = value if isinstance(value, (int, float)) else value
    if isinstance(value, (int, float)):
        source = full_like_from_tensor(input, float(value))
    out_pos = _scatter_flat_positions(input._shape, dim, index)
    out = scatter_from_tensor(input.reshape(-1), 0, out_pos, source.reshape(-1))
    out = out.reshape(list(input._shape))
    if is_grad_enabled() and input._requires_grad:
        out._requires_grad = True
        parents = [input]
        out._node = _Node(out, lambda g: _grad_scatter(g, input, dim, index, source), parents)
    return out


def index_add_from_tensor(input: "Tensor", dim: int, index: "Tensor", source: "Tensor") -> "Tensor":
    """out = input; out.index_add_(dim, index, source) — accumulate scatter.

    Uses the GPU ``scatterAdd`` runtime op. For duplicate ``index`` entries this
    relies on the runtime's (racy) accumulation, matching the pre-existing
    scatter behaviour; the race-free variant is used by the autograd backward
    rules via ``scatter_add_safe_from_tensor``.
    """
    from .autograd import _Node, is_grad_enabled, _grad_scatter

    out_pos = _scatter_flat_positions(input._shape, dim, index)
    out = scatter_add_from_tensor(input.reshape(-1), 0, out_pos, source.reshape(-1))
    out = out.reshape(list(input._shape))
    if is_grad_enabled() and (input._requires_grad or source._requires_grad):
        out._requires_grad = True
        parents = [input, source]
        out._node = _Node(out, lambda g: _grad_scatter(g, input, dim, index, source), parents)
    return out


def take_from_tensor(input: "Tensor", index: "Tensor") -> "Tensor":
    """Flatten ``input`` then gather at ``index`` (equivalent to torch.take)."""
    from .autograd import _Node, is_grad_enabled, _grad_gather

    out = gather_from_tensor(input.reshape(-1), 0, index.reshape(-1))
    if is_grad_enabled() and input._requires_grad:
        out._requires_grad = True
        out._node = _Node(out, lambda g: (_grad_gather(g, input, 0, index.reshape(-1)),), [input])
    return out


def unfold_from_tensor(input: "Tensor", dimension: int, size: int, step: int = 1) -> "Tensor":
    """Sliding-window view along ``dimension`` (equivalent to torch.Tensor.unfold).

    Each window is extracted with ``narrow`` (GPU) and stacked; the window axis is
    inserted immediately after ``dimension``.
    """
    d = dimension if dimension >= 0 else dimension + len(input._shape)
    L = input._shape[d]
    num = 0 if L < size else (L - size) // step + 1
    if num == 0:
        new_shape = list(input._shape)
        new_shape[d] = size
        new_shape.insert(d + 1, 0)
        from .tensor_factories_ops import zeros_from_shape

        return zeros_from_shape(new_shape, input._dtype)
    windows = [input.narrow(d, w * step, size).unsqueeze(d) for w in range(num)]
    return cat_multi_from_tensors(windows, dim=d)


def cdist_from_tensor(x1: "Tensor", x2: "Tensor", p: float = 2.0) -> "Tensor":
    """Pairwise distance matrix between rows of x1 and x2 (equivalent to torch.cdist)."""
    x1_2d = len(x1._shape) == 2
    x2_2d = len(x2._shape) == 2
    a = x1.unsqueeze(0) if x1_2d else x1
    b = x2.unsqueeze(0) if x2_2d else x2
    diff = a.unsqueeze(2) - b.unsqueeze(1)  # [B, N, M, D]
    if p == 2.0 or p == 2:
        dist = diff.pow(2).sum(-1).sqrt()
    else:
        dist = diff.abs().pow(p).sum(-1).pow(1.0 / p)
    if x1_2d and x2_2d:
        dist = dist.squeeze(0)
    return dist


def pdist_from_tensor(input: "Tensor", p: float = 2.0) -> "Tensor":
    """Pairwise distances of a single set, returned as the upper-triangular vector.

    Computed on the GPU via the dedicated ``pdist`` shader (no CPU readback).
    """
    from ._tensor_runtime_bridge import pdist_from_tensor as _gpu_pdist

    return _gpu_pdist(input, p=p)


def cat_from_tensors(tensors: Sequence["Tensor"], dim: int = 0) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_cat

    runtime = _get_runtime()
    ids = [t._id for t in tensors]
    meta = _run_js_awaitable(runtime.cat(ids, int(dim)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and any(t._requires_grad for t in tensors):
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        parents = list(tensors)
        result._node = _Node(result, lambda g: _grad_cat(g, tensors, dim), parents)
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def cat_multi_from_tensors(tensors: Sequence["Tensor"], dim: int = 0) -> "Tensor":
    """Concatenate any number of tensors (the runtime cat supports two at a time)."""
    result = tensors[0]
    for t in tensors[1:]:
        result = cat_from_tensors([result, t], dim=dim)
    return result


def stack_from_tensors(tensors: Sequence["Tensor"], dim: int = 0) -> "Tensor":
    if len(tensors) == 0:
        raise ValueError("stack requires at least one tensor")
    from .__init__ import cat
    unsqueezed = [t.unsqueeze(dim) for t in tensors]
    return cat(unsqueezed, dim=dim)


def searchsorted_from_tensor(
    sorted_sequence: "Tensor", values: "Tensor", right: bool = False
) -> "Tensor":
    """Return insertion indices of ``values`` into ``sorted_sequence`` (1-D)."""
    s = sorted_sequence.reshape(-1)
    v = values.reshape(-1)
    s_exp = s.unsqueeze(1)
    v_exp = v.unsqueeze(0)
    cmp = s_exp.le(v_exp) if right else s_exp.lt(v_exp)
    res = cmp.to(s._dtype).sum(0)
    return res.reshape(values._shape) if len(values._shape) > 1 else res


def kthvalue_from_tensor(x: "Tensor", k: int, dim: int = -1):
    from ._tensor import Tensor
    d = dim if dim >= 0 else dim + len(x._shape)
    vals, idx = sort_from_tensor(x, d, descending=False)
    out_val = vals.select(d, k - 1)
    out_idx = idx.select(d, k - 1)
    return out_val, out_idx


def median_from_tensor(x: "Tensor", dim=None):
    if dim is None:
        flat = x.reshape(-1)
        n = flat._shape[0]
        k = (n + 1) // 2
        out, _ = kthvalue_from_tensor(flat, k, dim=0)
        return out
    d = dim if dim >= 0 else dim + len(x._shape)
    n = x._shape[d]
    k = (n + 1) // 2
    out, _ = kthvalue_from_tensor(x, k, dim=d)
    return out


def quantile_from_tensor(x: "Tensor", q, dim: int = -1):
    from .__init__ import tensor as _tensor
    d = dim if dim >= 0 else dim + len(x._shape)
    vals, _ = sort_from_tensor(x, d, descending=False)
    n = x._shape[d]
    qs = q.tolist() if hasattr(q, "tolist") else (q if isinstance(q, (list, tuple)) else [q])
    results = []
    for qval in qs:
        pos = (n - 1) * qval
        k0 = int(_tensor([pos]).floor().item())
        k0 = max(0, min(n - 1, k0))
        k1 = min(n - 1, k0 + 1)
        w = pos - k0
        v0 = vals.select(d, k0)
        v1 = vals.select(d, k1)
        if k0 == k1:
            results.append(v0)
        else:
            results.append(v0.lerp(v1, w))
    if len(results) == 1:
        return results[0]
    from .__init__ import stack
    return stack(results)


def _mode_2d(xt: "Tensor"):
    """Mode per row of a 2-D tensor [rows, n]; returns (mode_values, counts)."""
    from ._tensor import Tensor
    from .tensor_factories_ops import arange_from_values, zeros_from_shape, ones_from_shape
    n = xt._shape[1]
    rows = xt._shape[0]
    vals, _ = sort_from_tensor(xt, 1, descending=False)
    shifted = roll_from_tensor(vals, 1, 1)
    is_first = vals.ne(shifted)
    first_col = arange_from_values(0, n, 1).unsqueeze(0).expand([rows, n])
    first_col = first_col.eq(zeros_from_shape([rows, n], first_col._dtype))
    is_first = is_first.logical_or(first_col)
    group_id = cumsum_from_tensor(is_first.to(vals._dtype), 1)
    last_gid = group_id.select(1, n - 1)
    ng = int(last_gid.max().item())
    row_ids = arange_from_values(0, rows, 1).unsqueeze(1).expand_as(group_id)
    global_gid = (row_ids * ng + (group_id - 1)).to("int64")
    ones = ones_from_shape([rows, n], vals._dtype)
    sizes = scatter_add_safe_from_tensor(
        zeros_from_shape([rows * ng], vals._dtype),
        0,
        global_gid.reshape(-1).to("int64"),
        ones.reshape(-1),
    )
    sizes = sizes.reshape([rows, ng])
    mode_g = sizes.argmax(1)
    cs = sizes.cumsum(1)
    end_pos = cs.gather(1, mode_g.unsqueeze(1))
    start_pos = end_pos - sizes.gather(1, mode_g.unsqueeze(1))
    mode_vals = vals.gather(1, start_pos.to("int64")).squeeze(1)
    mode_cnt = sizes.gather(1, mode_g.unsqueeze(1)).squeeze(1)
    return mode_vals, mode_cnt


def mode_from_tensor(x: "Tensor", dim=None):
    if dim is None:
        flat = x.reshape(-1)
        return _mode_2d(flat.unsqueeze(0))
    d = dim if dim >= 0 else dim + len(x._shape)
    perm = [i for i in range(len(x._shape)) if i != d] + [d]
    xt = x.permute(perm)
    other = 1
    for s in xt._shape[:-1]:
        other *= s
    rows = other if other > 0 else 1
    xt2 = xt.reshape([rows, x._shape[d]])
    vals, cnts = _mode_2d(xt2)
    out_shape = [s for i, s in enumerate(x._shape) if i != d]
    return vals.reshape(out_shape), cnts.reshape(out_shape)


def unique_from_tensor(x: "Tensor", return_counts: bool = False, sorted: bool = True, dim=None):
    if dim is not None:
        raise NotImplementedError("unique along a dimension is not supported yet")
    flat = x.reshape(-1)
    n = flat._shape[0]
    from .tensor_factories_ops import arange_from_values, zeros_from_shape, ones_from_shape
    vals, _ = sort_from_tensor(flat, 0, descending=False)
    shifted = roll_from_tensor(vals, 1, 0)
    is_first = vals.ne(shifted)
    first_pos = arange_from_values(0, n, 1)
    first_pos = first_pos.eq(zeros_from_shape([n], first_pos._dtype))
    is_first = is_first.logical_or(first_pos)
    uniq = vals.masked_select(is_first)
    if return_counts:
        group_id = is_first.cumsum(0)
        ng = int(group_id.select(0, n - 1).item())
        ones = ones_from_shape([n], vals._dtype)
        counts = scatter_add_safe_from_tensor(
            zeros_from_shape([ng], vals._dtype),
            0,
            (group_id - 1).to("int64"),
            ones,
        )
        return uniq, counts
    return uniq


def histogram_from_tensor(x: "Tensor", bins: int, range=None):
    from ._tensor import Tensor
    from .tensor_factories_ops import arange_from_values, zeros_from_shape, ones_from_shape
    flat = x.reshape(-1)
    if range is None:
        lo = flat.min()
        hi = flat.max()
    else:
        lo, hi = range[0], range[1]
    width = (hi - lo) / bins
    idx = ((flat - lo) / width).floor().clamp(0, bins - 1).to("int64")
    ones = ones_from_shape([flat._shape[0]], flat._dtype)
    counts = scatter_add_safe_from_tensor(
        zeros_from_shape([bins], flat._dtype), 0, idx, ones
    )
    edges = arange_from_values(0, bins + 1, 1).mul(width).add(lo)
    return counts, edges


def expand_from_tensor(tensor: "Tensor", shape: int | Sequence[int]) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_expand
    from .tensor_shape_utils import _normalize_shape

    runtime = _get_runtime()
    normalized = _normalize_shape(shape)
    meta = _run_js_awaitable(runtime.expand(tensor._id, normalized))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_expand(g, tensor, normalized),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def index_select_from_tensor(input: "Tensor", dim: int, index: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_index_select

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.indexSelect(input._id, int(dim), index._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and input._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_index_select(g, input, dim, index),), [input])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def sigmoid_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_sigmoid

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.sigmoid(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_sigmoid(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def tanh_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_tanh

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.tanh(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_tanh(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def sin_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.sin(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def cos_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.cos(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def gelu_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_gelu

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.gelu(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_gelu(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def silu_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_silu

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.silu(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_silu(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def leaky_relu_from_tensor(tensor: "Tensor", alpha: float = 0.01) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_leaky_relu

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.leakyRelu(tensor._id, alpha))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_leaky_relu(g, tensor, alpha),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def floor_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.floor(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def ceil_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.ceil(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def round_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.round(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def reciprocal_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.reciprocal(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def square_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.square(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def _make_unary_from_tensor(fn_name: str):
    def wrapper(tensor: "Tensor") -> "Tensor":
        from ._tensor import Tensor
        runtime = _get_runtime()
        meta = _run_js_awaitable(getattr(runtime, fn_name)(tensor._id))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)
    wrapper.__name__ = fn_name + "_from_tensor"
    return wrapper


tan_from_tensor = _make_unary_from_tensor("tan")
asin_from_tensor = _make_unary_from_tensor("asin")
acos_from_tensor = _make_unary_from_tensor("acos")
atan_from_tensor = _make_unary_from_tensor("atan")
sinh_from_tensor = _make_unary_from_tensor("sinh")
cosh_from_tensor = _make_unary_from_tensor("cosh")
asinh_from_tensor = _make_unary_from_tensor("asinh")
acosh_from_tensor = _make_unary_from_tensor("acosh")
atanh_from_tensor = _make_unary_from_tensor("atanh")
exp2_from_tensor = _make_unary_from_tensor("exp2")
log2_from_tensor = _make_unary_from_tensor("log2")
log10_from_tensor = _make_unary_from_tensor("log10")
log1p_from_tensor = _make_unary_from_tensor("log1p")
expm1_from_tensor = _make_unary_from_tensor("expm1")
trunc_from_tensor = _make_unary_from_tensor("trunc")
frac_from_tensor = _make_unary_from_tensor("frac")
softplus_from_tensor = _make_unary_from_tensor("softplus")
mish_from_tensor = _make_unary_from_tensor("mish")
hardsigmoid_from_tensor = _make_unary_from_tensor("hardsigmoid")
hardswish_from_tensor = _make_unary_from_tensor("hardswish")
softsign_from_tensor = _make_unary_from_tensor("softsign")
tanhshrink_from_tensor = _make_unary_from_tensor("tanhshrink")
rsqrt_from_tensor = _make_unary_from_tensor("rsqrt")
sign_from_tensor = _make_unary_from_tensor("sign")
sgn_from_tensor = _make_unary_from_tensor("sgn")
isnan_from_tensor = _make_unary_from_tensor("isnan")
isinf_from_tensor = _make_unary_from_tensor("isinf")
isfinite_from_tensor = _make_unary_from_tensor("isfinite")
isposinf_from_tensor = _make_unary_from_tensor("isposinf")
isneginf_from_tensor = _make_unary_from_tensor("isneginf")
logical_not_from_tensor = _make_unary_from_tensor("logicalNot")
erf_from_tensor = _make_unary_from_tensor("erf")
erfc_from_tensor = _make_unary_from_tensor("erfc")
lgamma_from_tensor = _make_unary_from_tensor("lgamma")
digamma_from_tensor = _make_unary_from_tensor("digamma")
i0_from_tensor = _make_unary_from_tensor("i0")
deg2rad_from_tensor = _make_unary_from_tensor("deg2rad")
rad2deg_from_tensor = _make_unary_from_tensor("rad2deg")


def relu_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_relu

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.relu(tensor._id))
    tensor_id, shape, dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, shape, dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_relu(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, shape, dtype)


def abs_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_abs

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.abs(tensor._id))
    tensor_id, shape, dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, shape, dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_abs(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, shape, dtype)


def sqrt_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_sqrt

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.sqrt(tensor._id))
    tensor_id, shape, dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, shape, dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_sqrt(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, shape, dtype)


def exp_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_exp

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.exp(tensor._id))
    tensor_id, shape, dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, shape, dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_exp(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, shape, dtype)


def log_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_log

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.log(tensor._id))
    tensor_id, shape, dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, shape, dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_log(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, shape, dtype)


def neg_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_neg

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.neg(tensor._id))
    tensor_id, shape, dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, shape, dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_neg(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, shape, dtype)


def select_from_tensor(tensor: "Tensor", dim: int, index: int) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_select

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.select(tensor._id, int(dim), int(index)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_select(g, tensor, dim, index),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def slice_from_tensor(tensor: "Tensor", dim: int, start: int | None = None, end: int | None = None, step: int = 1) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_slice

    runtime = _get_runtime()
    if start is None and end is None:
        meta = _run_js_awaitable(runtime.slice(tensor._id, int(dim), None, None, int(step)))
    elif end is None:
        meta = _run_js_awaitable(runtime.slice(tensor._id, int(dim), int(start), None, int(step)))
    else:
        meta = _run_js_awaitable(runtime.slice(tensor._id, int(dim), int(start) if start is not None else None, int(end), int(step)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        actual_start = start if start is not None else 0
        actual_step = step
        result._node = _Node(result, lambda g: (_grad_slice(g, tensor, dim, actual_start, end, actual_step),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def matmul_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_matmul

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.matmul(a._id, b._id))
    tensor_id, shape, dtype = _js_meta_to_tuple(meta)

    requires_grad = is_grad_enabled() and (a._requires_grad or b._requires_grad)
    if requires_grad:
        result_tensor = Tensor(tensor_id, shape, dtype, _requires_grad=True)
        node = _Node(
            tensor=result_tensor,
            grad_fn=lambda grad_out: _grad_matmul(grad_out, a, b),
            parents=[a, b],
        )
        result_tensor._node = node
        return result_tensor

    return Tensor(tensor_id, shape, dtype)


def sum_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_sum

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.sum(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_sum(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def sum_dim_from_tensor(tensor: "Tensor", dim: int, keepdim: bool = False) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_sum_dim

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.sumDim(tensor._id, int(dim), keepdim))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_sum_dim(g, tensor, dim, keepdim),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def mean_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_mean

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.mean(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_mean(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def mean_dim_from_tensor(tensor: "Tensor", dim: int, keepdim: bool = False) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_mean_dim

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.meanDim(tensor._id, int(dim), keepdim))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_mean_dim(g, tensor, dim, keepdim),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def prod_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_prod

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.prod(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_prod(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def min_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_min

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.min(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_min(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def max_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_max

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.max(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_max(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def max_dim_from_tensor(tensor: "Tensor", dim: int, keepdim: bool = False) -> tuple["Tensor", "Tensor"]:
    d = dim if dim >= 0 else dim + len(tensor._shape)
    values, indices = topk_from_tensor(tensor, 1, dim=d, largest=True)
    if not keepdim:
        values = values.squeeze(d)
        indices = indices.squeeze(d)
    return values, indices


def min_dim_from_tensor(tensor: "Tensor", dim: int, keepdim: bool = False) -> tuple["Tensor", "Tensor"]:
    d = dim if dim >= 0 else dim + len(tensor._shape)
    values, indices = topk_from_tensor(tensor, 1, dim=d, largest=False)
    if not keepdim:
        values = values.squeeze(d)
        indices = indices.squeeze(d)
    return values, indices


def _amax_amin_from_tensor(tensor: "Tensor", dim, keepdim: bool, largest: bool) -> "Tensor":
    ndim = len(tensor._shape)
    if dim is None:
        dims = list(range(ndim))
    elif isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)
    dims = sorted((d if d >= 0 else d + ndim) for d in dims)
    result = tensor
    for d in reversed(dims):
        if largest:
            result, _ = max_dim_from_tensor(result, d, keepdim=True)
        else:
            result, _ = min_dim_from_tensor(result, d, keepdim=True)
    if not keepdim:
        for d in reversed(dims):
            result = result.squeeze(d)
    return result


def amax_from_tensor(tensor: "Tensor", dim=None, keepdim: bool = False) -> "Tensor":
    return _amax_amin_from_tensor(tensor, dim, keepdim, largest=True)


def amin_from_tensor(tensor: "Tensor", dim=None, keepdim: bool = False) -> "Tensor":
    return _amax_amin_from_tensor(tensor, dim, keepdim, largest=False)


def var_from_tensor(tensor: "Tensor", dim=None, keepdim: bool = False, correction: int = 1) -> "Tensor":
    ndim = len(tensor._shape)
    if dim is None:
        dims = list(range(ndim))
    elif isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)
    dims = sorted((d if d >= 0 else d + ndim) for d in dims)
    n = 1
    for d in dims:
        n *= tensor._shape[d]
    mu = tensor
    for d in dims:
        mu = mean_dim_from_tensor(mu, d, keepdim=True)
    centered = tensor.sub(mu)
    s = centered.mul(centered)
    for d in dims:
        s = sum_dim_from_tensor(s, d, keepdim=True)
    denom = n - correction
    result = s.mul(1.0 / denom) if denom > 0 else s.mul(float("inf"))
    if not keepdim:
        for d in reversed(dims):
            result = result.squeeze(d)
    return result


def nan_to_num_from_tensor(tensor: "Tensor", nan: float = 0.0, posinf=None, neginf=None) -> "Tensor":
    import torch as _torch

    dt = tensor._dtype
    finfo_max = 3.4028234663852886e38
    pinf = finfo_max if posinf is None else float(posinf)
    ninf = -finfo_max if neginf is None else float(neginf)
    nan_t = _torch.full_like(tensor, nan, dtype=dt)
    pos_t = _torch.full_like(tensor, pinf, dtype=dt)
    neg_t = _torch.full_like(tensor, ninf, dtype=dt)
    out = nan_t.where(tensor.isnan(), tensor)
    out = pos_t.where(tensor.isposinf(), out)
    out = neg_t.where(tensor.isneginf(), out)
    return out


def movedim_from_tensor(tensor: "Tensor", source, destination) -> "Tensor":
    ndim = len(tensor._shape)
    srcs = [source] if isinstance(source, int) else list(source)
    dsts = [destination] if isinstance(destination, int) else list(destination)
    srcs = [s if s >= 0 else s + ndim for s in srcs]
    dsts = [d if d >= 0 else d + ndim for d in dsts]
    order = [d for d in range(ndim) if d not in srcs]
    for dst, src in sorted(zip(dsts, srcs)):
        order.insert(dst, src)
    return tensor.permute(order)


def masked_select_from_tensor(tensor: "Tensor", mask: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_masked_select

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.maskedSelect(tensor._id, mask._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_masked_select(g, tensor, mask),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def masked_fill_from_tensor(tensor: "Tensor", mask: "Tensor", value: float) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_masked_fill

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.maskedFill(tensor._id, mask._id, float(value)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_masked_fill(g, tensor, mask, value),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def nonzero_from_tensor(tensor: "Tensor") -> "Tensor":
    """Returns indices of non-zero elements as a 2D tensor of shape [N, ndim].

    Each row contains the coordinates of a non-zero element.
    """
    from ._tensor import Tensor
    from .__init__ import cat

    runtime = _get_runtime()
    result = _run_js_awaitable(runtime.nonzero(tensor._id))
    count = int(result.count)
    linear_id = int(result.indices.id)
    linear_shape = list(result.indices.shape.to_py() if hasattr(result.indices.shape, "to_py") else result.indices.shape)
    linear_dtype = str(result.indices.dtype)
    ndim = len(tensor.shape)

    if count == 0:
        from .tensor_factories_ops import full_from_shape
        return full_from_shape([0, ndim], 0, dtype="int64")

    # Convert linear indices to coordinate tuples.
    flat_indices = Tensor(linear_id, linear_shape, linear_dtype)
    strides = [1]
    for s in reversed(tensor.shape[1:]):
        strides.insert(0, strides[0] * s)
    coords = []
    for dim_idx, stride in enumerate(strides):
        coord = flat_indices.floor_divide(stride).remainder(tensor.shape[dim_idx])
        coords.append(coord.to(dtype="int64"))
    return cat(coords, dim=1).to(dtype="int64")


def roll_from_tensor(tensor: "Tensor", shifts: int | list[int], dims: int | list[int] | None = None) -> "Tensor":
    """Roll the tensor along given dimensions.

    Flat shift if no dims specified. Multi-dim roll applied sequentially.
    """
    from ._tensor import Tensor
    from .__init__ import cat
    if dims is None:
        shift = int(shifts) if isinstance(shifts, (int, float)) else int(shifts[0])
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.roll(tensor._id, shift))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)
    shifts_list = [shifts] if isinstance(shifts, int) else list(shifts)
    dims_list = [dims] if isinstance(dims, int) else list(dims)
    result = tensor
    for s, d in zip(shifts_list, dims_list):
        d_norm = d if d >= 0 else d + len(tensor.shape)
        n = tensor.shape[d_norm]
        shift_mod = int(s) % n
        if shift_mod == 0:
            continue
        front = result.narrow(d_norm, 0, n - shift_mod)
        back = result.narrow(d_norm, n - shift_mod, shift_mod)
        result = cat([back, front], dim=d_norm)
    return result


def softmax_from_tensor(tensor: "Tensor", dim: int = -1) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_softmax

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.softmax(tensor._id, int(dim)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g, out=result: (_grad_softmax(g, tensor, dim, out),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def log_softmax_from_tensor(tensor: "Tensor", dim: int = -1) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_log_softmax

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.logSoftmax(tensor._id, int(dim)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_log_softmax(g, tensor, dim),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def eq_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.eq(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def ne_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.ne(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def lt_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.lt(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def le_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.le(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def gt_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.gt(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def ge_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.ge(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def equal_from_tensors(a: "Tensor", b: "Tensor") -> "Tensor":
    """Returns True if two tensors are element-wise equal."""
    return a.eq(b).all()


def isclose_from_tensors(
    a: "Tensor",
    b: "Tensor",
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> "Tensor":
    """Returns a boolean tensor where two tensors are element-wise close."""
    from .__init__ import abs, le, logical_or, logical_and, ne
    # |a - b| <= atol + rtol * |b|
    diff = abs(a.sub(b))
    rhs = abs(b).mul(rtol).add(atol)
    result = le(diff, rhs)
    if equal_nan:
        both_nan = logical_and(ne(a, a), ne(b, b))
        result = logical_or(result, both_nan)
    return result


def allclose_from_tensors(
    a: "Tensor",
    b: "Tensor",
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> "Tensor":
    """Returns True if all elements are close."""
    from .__init__ import all
    return all(isclose_from_tensors(a, b, rtol, atol, equal_nan))
