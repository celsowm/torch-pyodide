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


def all_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.all(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def cumsum_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_cumsum

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.cumsum(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_cumsum(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def cumprod_from_tensor(tensor: "Tensor") -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_cumprod

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.cumprod(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_cumprod(g, tensor),), [tensor])
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
    from .tensor_factories_ops import tensor_from_data

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    d = dim if dim >= 0 else dim + len(tensor._shape)
    size = tensor._shape[d]
    if k >= size:
        return tensor, tensor_from_data(list(range(size)), list(tensor._shape), "int64")
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
    from .autograd import _Node, is_grad_enabled, _grad_scatter
    from .tensor_shape_utils import _flatten_out
    from .tensor_factories_ops import tensor_from_data

    result = tensor.clone()
    flat = result.tolist()
    flat_list: list[float] = _flatten_out(flat)
    idx = index.tolist()
    idx_flat: list[float] = _flatten_out(idx)
    out_len = len(flat_list)
    if isinstance(src, (int, float)):
        val = float(src)
        for i in range(len(idx_flat)):
            pos = int(idx_flat[i])
            if 0 <= pos < out_len:
                flat_list[pos] = val
    else:
        src_flat = _flatten_out(src.tolist())
        for i in range(min(len(idx_flat), len(src_flat))):
            pos = int(idx_flat[i])
            if 0 <= pos < out_len:
                flat_list[pos] = src_flat[i]
    out = tensor_from_data(flat_list, list(tensor._shape), tensor._dtype)

    if is_grad_enabled() and (tensor._requires_grad or (not isinstance(src, (int, float)) and src._requires_grad)):
        out._requires_grad = True
        parents = [tensor]
        if not isinstance(src, (int, float)):
            parents.append(src)
        out._node = _Node(out, lambda g: _grad_scatter(g, tensor, dim, index, src), parents)
    return out


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


def stack_from_tensors(tensors: Sequence["Tensor"], dim: int = 0) -> "Tensor":
    if len(tensors) == 0:
        raise ValueError("stack requires at least one tensor")
    from .__init__ import cat
    unsqueezed = [t.unsqueeze(dim) for t in tensors]
    return cat(unsqueezed, dim=dim)


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
    from .autograd import _Node, is_grad_enabled, _grad_sum

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.sumDim(tensor._id, int(dim), keepdim))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_sum(g, tensor),), [tensor])
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
    from .autograd import _Node, is_grad_enabled, _grad_mean

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.meanDim(tensor._id, int(dim), keepdim))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_mean(g, tensor),), [tensor])
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
