from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence

from ._runtime import _get_runtime, _run_js_awaitable

if TYPE_CHECKING:
    from .autograd import _Node


@dataclass(slots=True)
class Tensor:
    _id: int
    _shape: list[int]
    _dtype: str
    _requires_grad: bool = False
    _node: "_Node | None" = field(default=None, repr=False)
    _backward_hooks: dict[int, Callable] = field(default_factory=dict, repr=False)
    grad: "Tensor | None" = field(default=None, repr=False)
    _retains_grad: bool = field(default=False, repr=False)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._shape)

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def numel(self) -> int:
        n = 1
        for s in self._shape:
            n *= s
        return n

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = value

    def requires_grad_(self, requires_grad_: bool = True) -> "Tensor":
        self._requires_grad = requires_grad_
        return self

    @property
    def is_leaf(self) -> bool:
        return self._node is None

    def register_hook(self, hook: Callable[["Tensor"], "Tensor | None"]) -> int:
        hook_id = id(hook)
        self._backward_hooks[hook_id] = hook
        return hook_id

    def remove_hook(self, hook_id: int) -> None:
        self._backward_hooks.pop(hook_id, None)

    def retain_grad(self) -> None:
        self._retains_grad = True

    def backward(
        self,
        gradient: "Tensor | None" = None,
        retain_graph: bool = False,
        create_graph: bool = False,
        inputs: Sequence["Tensor"] | None = None,
    ) -> None:
        from .autograd import _backward_from_tensor
        _backward_from_tensor(self, gradient, retain_graph, create_graph)

    def to(self, dtype: str, device: object = None) -> "Tensor":
        if dtype is not None and dtype != self._dtype:
            from .tensor_factories_ops import zeros_like_from_tensor
            from .tensor_ops import add_from_tensors
            empty = zeros_like_from_tensor(self, dtype)
            return add_from_tensors(empty, self)
        return self

    def clone(self) -> "Tensor":
        return self.add(0.0)

    def detach(self) -> "Tensor":
        t = Tensor(self._id, list(self._shape), self._dtype)
        return t

    def detach_(self) -> "Tensor":
        self._node = None
        self.grad = None
        return self

    def contiguous(self) -> "Tensor":
        return self

    cpu = detach
    cuda = detach

    def half(self) -> "Tensor":
        return Tensor(self._id, self._shape, "float16", _requires_grad=self._requires_grad)

    def bfloat16(self) -> "Tensor":
        return Tensor(self._id, self._shape, "bfloat16", _requires_grad=self._requires_grad)

    def float(self) -> "Tensor":
        return self if self._dtype == "float32" else self.to("float32")

    def double(self) -> "Tensor":
        return self.to("float64") if self._dtype != "float64" else self

    def long(self) -> "Tensor":
        return self.to("int64")

    def int(self) -> "Tensor":
        return self.to("int32")

    def byte(self) -> "Tensor":
        return self.to("uint8")

    def bool(self) -> "Tensor":
        return self.to("bool")

    def __bool__(self) -> bool:
        return self.item() != 0

    def __len__(self) -> int:
        return self._shape[0] if self._shape else 0

    def zero_(self) -> "Tensor":
        return self

    def fill_(self, value: float) -> "Tensor":
        return self

    def copy_(self, src: "Tensor") -> "Tensor":
        return src

    def narrow(self, dim: int, start: int, length: int) -> "Tensor":
        return self.slice(dim=dim, start=start, end=start + length)

    def repeat(self, *sizes: int) -> "Tensor":
        from ._tensor_runtime_bridge import repeat_from_tensor
        return repeat_from_tensor(self, list(sizes))

    def repeat_interleave(self, repeats: int, dim: int | None = None) -> "Tensor":
        from ._tensor_math_helpers import repeat_interleave_from_tensor
        return repeat_interleave_from_tensor(self, repeats, dim)

    tile = repeat

    def topk(self, k: int, dim: int = -1, largest: bool = True) -> tuple["Tensor", "Tensor"]:
        from .tensor_ops import topk_from_tensor
        return topk_from_tensor(self, k, dim, largest)

    def sort(self, dim: int = -1, descending: bool = False) -> tuple["Tensor", "Tensor"]:
        from .tensor_ops import sort_from_tensor
        return sort_from_tensor(self, dim, descending)

    def gather(self, dim: int, index: "Tensor") -> "Tensor":
        from .tensor_ops import gather_from_tensor
        return gather_from_tensor(self, dim, index)

    def scatter_(self, dim: int, index: "Tensor", src: "Tensor | float") -> "Tensor":
        from .tensor_ops import scatter_from_tensor
        return scatter_from_tensor(self, dim, index, src)

    def argsort(self, dim: int = -1, descending: bool = False) -> "Tensor":
        from .tensor_ops import sort_from_tensor
        _, indices = sort_from_tensor(self, dim, descending)
        return indices

    def __add__(self, other: "Tensor | float") -> "Tensor":
        return self.add(other)

    def __radd__(self, other: "Tensor | float") -> "Tensor":
        from ._tensor_math_helpers import radd_from_tensor
        return radd_from_tensor(self, other)

    def __mul__(self, other: "Tensor | float") -> "Tensor":
        return self.mul(other)

    def __rmul__(self, other: "Tensor | float") -> "Tensor":
        return self.mul(other)

    def __sub__(self, other: "Tensor | float") -> "Tensor":
        return self.sub(other)

    def __rsub__(self, other: "Tensor | float") -> "Tensor":
        from ._tensor_math_helpers import rsub_from_tensor
        return rsub_from_tensor(self, other)

    def __truediv__(self, other: "Tensor | float") -> "Tensor":
        return self.div(other)

    def __neg__(self) -> "Tensor":
        return self.neg()

    def __lt__(self, other: "Tensor | float | int | bool") -> "Tensor":
        from .tensor_ops import lt_from_tensors, _scalar_to_tensor
        if isinstance(other, (float, int, bool)):
            other = _scalar_to_tensor(other, self._dtype)
        return lt_from_tensors(self, other)

    def __le__(self, other: "Tensor | float | int | bool") -> "Tensor":
        from .tensor_ops import le_from_tensors, _scalar_to_tensor
        if isinstance(other, (float, int, bool)):
            other = _scalar_to_tensor(other, self._dtype)
        return le_from_tensors(self, other)

    def __gt__(self, other: "Tensor | float | int | bool") -> "Tensor":
        from .tensor_ops import gt_from_tensors, _scalar_to_tensor
        if isinstance(other, (float, int, bool)):
            other = _scalar_to_tensor(other, self._dtype)
        return gt_from_tensors(self, other)

    def __ge__(self, other: "Tensor | float | int | bool") -> "Tensor":
        from .tensor_ops import ge_from_tensors, _scalar_to_tensor
        if isinstance(other, (float, int, bool)):
            other = _scalar_to_tensor(other, self._dtype)
        return ge_from_tensors(self, other)

    def __eq__(self, other: "Tensor | float | int | bool") -> "Tensor":
        from .tensor_ops import eq_from_tensors, _scalar_to_tensor
        if isinstance(other, (float, int, bool)):
            other = _scalar_to_tensor(other, self._dtype)
        return eq_from_tensors(self, other)

    def __ne__(self, other: "Tensor | float | int | bool") -> "Tensor":
        from .tensor_ops import ne_from_tensors, _scalar_to_tensor
        if isinstance(other, (float, int, bool)):
            other = _scalar_to_tensor(other, self._dtype)
        return ne_from_tensors(self, other)

    def __pow__(self, other: "Tensor | float") -> "Tensor":
        from .tensor_ops import pow_from_tensors, _scalar_to_tensor
        return pow_from_tensors(self, other) if isinstance(other, Tensor) else pow_from_tensors(self, _scalar_to_tensor(float(other), self._dtype))

    def __and__(self, other: "Tensor") -> "Tensor":
        return self.mul(other)

    def __or__(self, other: "Tensor") -> "Tensor":
        return self.add(other).clamp(0.0, 1.0)

    def __xor__(self, other: "Tensor") -> "Tensor":
        return (self.__or__(other)).__sub__(self.__and__(other))

    def __invert__(self) -> "Tensor":
        from ._tensor_math_helpers import invert_from_tensor
        return invert_from_tensor(self)

    def matmul(self, other: "Tensor") -> "Tensor":
        from .tensor_ops import matmul_from_tensors
        return matmul_from_tensors(self, other)

    def mm(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    def bmm(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    def mv(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    def dot(self, other: "Tensor") -> "Tensor":
        return (self * other).sum()

    def outer(self, other: "Tensor") -> "Tensor":
        return self.reshape(-1, 1) * other.reshape(1, -1)

    def norm(self, p: float | str = "fro") -> "Tensor":
        from ._tensor_math_helpers import norm_from_tensor
        return norm_from_tensor(self, p)

    def cholesky(self) -> "Tensor":
        from ._tensor_runtime_bridge import cholesky_from_tensor
        return cholesky_from_tensor(self)

    def add(self, other: "Tensor | float") -> "Tensor":
        from .tensor_ops import add_from_tensors, _scalar_to_tensor
        b_tensor = other if isinstance(other, Tensor) else _scalar_to_tensor(float(other), self._dtype)
        return add_from_tensors(self, b_tensor)

    def sub(self, other: "Tensor | float") -> "Tensor":
        from .tensor_ops import sub_from_tensors, _scalar_to_tensor
        b_tensor = other if isinstance(other, Tensor) else _scalar_to_tensor(float(other), self._dtype)
        return sub_from_tensors(self, b_tensor)

    def mul(self, other: "Tensor | float") -> "Tensor":
        from .tensor_ops import mul_from_tensors, _scalar_to_tensor
        b_tensor = other if isinstance(other, Tensor) else _scalar_to_tensor(float(other), self._dtype)
        return mul_from_tensors(self, b_tensor)

    def div(self, other: "Tensor | float") -> "Tensor":
        from .tensor_ops import div_from_tensors, _scalar_to_tensor
        b_tensor = other if isinstance(other, Tensor) else _scalar_to_tensor(float(other), self._dtype)
        return div_from_tensors(self, b_tensor)

    def lu(self) -> tuple["Tensor", "Tensor"]:
        from ._tensor_runtime_bridge import lu_from_tensor
        return lu_from_tensor(self)

    def triangular_solve(self, b: "Tensor", upper: bool = False) -> "Tensor":
        from ._tensor_runtime_bridge import triangular_solve_from_tensors
        return triangular_solve_from_tensors(self, b, upper)

    def item(self) -> float:
        from ._tensor_runtime_bridge import item_from_tensor
        return item_from_tensor(self)

    def det(self) -> "Tensor":
        from ._tensor_linalg_py import det_from_tensor
        return det_from_tensor(self)

    def inv(self) -> "Tensor":
        from ._tensor_linalg_py import inv_from_tensor
        return inv_from_tensor(self)

    def diag(self) -> "Tensor":
        from ._tensor_linalg_py import diag_from_tensor
        return diag_from_tensor(self)

    def pow(self, other: "Tensor | float") -> "Tensor":
        from .tensor_ops import pow_from_tensors, _scalar_to_tensor
        return pow_from_tensors(self, other) if isinstance(other, Tensor) else pow_from_tensors(self, _scalar_to_tensor(float(other), self._dtype))

    def heaviside(self, values: "Tensor") -> "Tensor":
        from .tensor_ops import heaviside_from_tensors
        return heaviside_from_tensors(self, values)

    def maximum(self, other: "Tensor") -> "Tensor":
        from .tensor_ops import maximum_from_tensors
        return maximum_from_tensors(self, other)

    def minimum(self, other: "Tensor") -> "Tensor":
        from .tensor_ops import minimum_from_tensors
        return minimum_from_tensors(self, other)

    def any(self) -> "Tensor":
        from .tensor_ops import any_from_tensor
        return any_from_tensor(self)

    def all(self) -> "Tensor":
        from .tensor_ops import all_from_tensor
        return all_from_tensor(self)

    def cumsum(self) -> "Tensor":
        from .tensor_ops import cumsum_from_tensor
        return cumsum_from_tensor(self)

    def cumprod(self) -> "Tensor":
        from .tensor_ops import cumprod_from_tensor
        return cumprod_from_tensor(self)

    def tril(self, diagonal: int = 0) -> "Tensor":
        from .tensor_ops import tril_from_tensor
        return tril_from_tensor(self, diagonal)

    def triu(self, diagonal: int = 0) -> "Tensor":
        from .tensor_ops import triu_from_tensor
        return triu_from_tensor(self, diagonal)

    def flip(self, dims: Sequence[int]) -> "Tensor":
        from .tensor_ops import flip_from_tensor
        return flip_from_tensor(self, dims)

    def where(self, condition: "Tensor", other: "Tensor") -> "Tensor":
        from .tensor_ops import where_from_tensors
        return where_from_tensors(condition, self, other)

    def cat(self, other: "Tensor", dim: int = 0) -> "Tensor":
        from .tensor_ops import cat_from_tensors
        return cat_from_tensors([self, other], dim)

    def stack(self, other: "Tensor", dim: int = 0) -> "Tensor":
        from .tensor_ops import stack_from_tensors
        return stack_from_tensors([self, other], dim)

    def index_select(self, dim: int, index: "Tensor") -> "Tensor":
        from .tensor_ops import index_select_from_tensor
        return index_select_from_tensor(self, dim, index)

    def empty_like(self) -> "Tensor":
        from .tensor_factories_ops import empty_like_from_tensor
        return empty_like_from_tensor(self)

    def zeros_like(self) -> "Tensor":
        from .tensor_factories_ops import zeros_like_from_tensor
        return zeros_like_from_tensor(self)

    def ones_like(self) -> "Tensor":
        from .tensor_factories_ops import ones_like_from_tensor
        return ones_like_from_tensor(self)

    def relu(self) -> "Tensor":
        from .tensor_ops import relu_from_tensor
        return relu_from_tensor(self)

    def abs(self) -> "Tensor":
        from .tensor_ops import abs_from_tensor
        return abs_from_tensor(self)

    def sqrt(self) -> "Tensor":
        from .tensor_ops import sqrt_from_tensor
        return sqrt_from_tensor(self)

    def exp(self) -> "Tensor":
        from .tensor_ops import exp_from_tensor
        return exp_from_tensor(self)

    def log(self) -> "Tensor":
        from .tensor_ops import log_from_tensor
        return log_from_tensor(self)

    def neg(self) -> "Tensor":
        from .tensor_ops import neg_from_tensor
        return neg_from_tensor(self)

    def clamp(self, min: float, max: float) -> "Tensor":
        from ._tensor_runtime_bridge import clamp_from_tensor
        return clamp_from_tensor(self, min, max)

    def argmax(self) -> "Tensor":
        from ._tensor_runtime_bridge import argmax_from_tensor
        return argmax_from_tensor(self)

    def argmin(self) -> "Tensor":
        from ._tensor_runtime_bridge import argmin_from_tensor
        return argmin_from_tensor(self)

    def reshape(self, shape: int | Sequence[int]) -> "Tensor":
        from ._tensor_runtime_bridge import reshape_from_tensor
        return reshape_from_tensor(self, shape)

    def view(self, *shape: int) -> "Tensor":
        from .tensor_shape_utils import _normalize_shape_from_args
        normalized = _normalize_shape_from_args(shape)
        return self.reshape(normalized)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        from ._tensor_runtime_bridge import flatten_from_tensor
        return flatten_from_tensor(self, start_dim, end_dim)

    def squeeze(self, dim: int | None = None) -> "Tensor":
        from ._tensor_runtime_bridge import squeeze_from_tensor
        return squeeze_from_tensor(self, dim)

    def unsqueeze(self, dim: int) -> "Tensor":
        from ._tensor_runtime_bridge import unsqueeze_from_tensor
        return unsqueeze_from_tensor(self, dim)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        from ._tensor_runtime_bridge import transpose_from_tensor
        return transpose_from_tensor(self, dim0, dim1)

    def permute(self, dims: Sequence[int]) -> "Tensor":
        from ._tensor_runtime_bridge import permute_from_tensor
        return permute_from_tensor(self, dims)

    def expand(self, *shape: int) -> "Tensor":
        from .tensor_ops import expand_from_tensor
        return expand_from_tensor(self, list(shape))

    def select(self, dim: int, index: int) -> "Tensor":
        from .tensor_ops import select_from_tensor
        return select_from_tensor(self, dim, index)

    def slice(self, dim: int, start: int | None = None, end: int | None = None, step: int = 1) -> "Tensor":
        from .tensor_ops import slice_from_tensor
        return slice_from_tensor(self, dim, start, end, step)

    @property
    def T(self) -> "Tensor":
        from ._tensor_runtime_bridge import t_from_tensor
        return t_from_tensor(self)

    def tolist(self) -> object:
        from ._tensor_runtime_bridge import tolist_from_tensor
        return tolist_from_tensor(self)

    def split(self, split_size: int | list[int], dim: int = 0) -> list["Tensor"]:
        from ._tensor_shape_ops import split_from_tensor
        return split_from_tensor(self, split_size, dim)

    def chunk(self, chunks: int, dim: int = 0) -> list["Tensor"]:
        from ._tensor_shape_ops import chunk_from_tensor
        return chunk_from_tensor(self, chunks, dim)

    def destroy(self) -> None:
        from ._tensor_runtime_bridge import destroy_tensor
        destroy_tensor(self)

    def sigmoid(self) -> "Tensor":
        from .tensor_ops import sigmoid_from_tensor
        return sigmoid_from_tensor(self)

    def tanh(self) -> "Tensor":
        from .tensor_ops import tanh_from_tensor
        return tanh_from_tensor(self)

    def sin(self) -> "Tensor":
        from .tensor_ops import sin_from_tensor
        return sin_from_tensor(self)

    def cos(self) -> "Tensor":
        from .tensor_ops import cos_from_tensor
        return cos_from_tensor(self)

    def gelu(self) -> "Tensor":
        from .tensor_ops import gelu_from_tensor
        return gelu_from_tensor(self)

    def silu(self) -> "Tensor":
        from .tensor_ops import silu_from_tensor
        return silu_from_tensor(self)

    def softmax(self, dim: int = -1) -> "Tensor":
        from .tensor_ops import softmax_from_tensor
        return softmax_from_tensor(self, dim)

    def log_softmax(self, dim: int = -1) -> "Tensor":
        from .tensor_ops import log_softmax_from_tensor
        return log_softmax_from_tensor(self, dim)

    def leaky_relu(self, alpha: float = 0.01) -> "Tensor":
        from .tensor_ops import leaky_relu_from_tensor
        return leaky_relu_from_tensor(self, alpha)

    def floor(self) -> "Tensor":
        from .tensor_ops import floor_from_tensor
        return floor_from_tensor(self)

    def ceil(self) -> "Tensor":
        from .tensor_ops import ceil_from_tensor
        return ceil_from_tensor(self)

    def round(self) -> "Tensor":
        from .tensor_ops import round_from_tensor
        return round_from_tensor(self)

    def reciprocal(self) -> "Tensor":
        from .tensor_ops import reciprocal_from_tensor
        return reciprocal_from_tensor(self)

    def square(self) -> "Tensor":
        from .tensor_ops import square_from_tensor
        return square_from_tensor(self)

    def tan(self) -> "Tensor":
        from .tensor_ops import tan_from_tensor
        return tan_from_tensor(self)

    def asin(self) -> "Tensor":
        from .tensor_ops import asin_from_tensor
        return asin_from_tensor(self)

    def acos(self) -> "Tensor":
        from .tensor_ops import acos_from_tensor
        return acos_from_tensor(self)

    def atan(self) -> "Tensor":
        from .tensor_ops import atan_from_tensor
        return atan_from_tensor(self)

    def sinh(self) -> "Tensor":
        from .tensor_ops import sinh_from_tensor
        return sinh_from_tensor(self)

    def cosh(self) -> "Tensor":
        from .tensor_ops import cosh_from_tensor
        return cosh_from_tensor(self)

    def asinh(self) -> "Tensor":
        from .tensor_ops import asinh_from_tensor
        return asinh_from_tensor(self)

    def acosh(self) -> "Tensor":
        from .tensor_ops import acosh_from_tensor
        return acosh_from_tensor(self)

    def atanh(self) -> "Tensor":
        from .tensor_ops import atanh_from_tensor
        return atanh_from_tensor(self)

    def exp2(self) -> "Tensor":
        from .tensor_ops import exp2_from_tensor
        return exp2_from_tensor(self)

    def log2(self) -> "Tensor":
        from .tensor_ops import log2_from_tensor
        return log2_from_tensor(self)

    def log10(self) -> "Tensor":
        from .tensor_ops import log10_from_tensor
        return log10_from_tensor(self)

    def log1p(self) -> "Tensor":
        from .tensor_ops import log1p_from_tensor
        return log1p_from_tensor(self)

    def expm1(self) -> "Tensor":
        from .tensor_ops import expm1_from_tensor
        return expm1_from_tensor(self)

    def trunc(self) -> "Tensor":
        from .tensor_ops import trunc_from_tensor
        return trunc_from_tensor(self)

    def frac(self) -> "Tensor":
        from .tensor_ops import frac_from_tensor
        return frac_from_tensor(self)

    def softplus(self) -> "Tensor":
        from .tensor_ops import softplus_from_tensor
        return softplus_from_tensor(self)

    def mish(self) -> "Tensor":
        from .tensor_ops import mish_from_tensor
        return mish_from_tensor(self)

    def hardsigmoid(self) -> "Tensor":
        from .tensor_ops import hardsigmoid_from_tensor
        return hardsigmoid_from_tensor(self)

    def hardswish(self) -> "Tensor":
        from .tensor_ops import hardswish_from_tensor
        return hardswish_from_tensor(self)

    def softsign(self) -> "Tensor":
        from .tensor_ops import softsign_from_tensor
        return softsign_from_tensor(self)

    def tanhshrink(self) -> "Tensor":
        from .tensor_ops import tanhshrink_from_tensor
        return tanhshrink_from_tensor(self)

    def rsqrt(self) -> "Tensor":
        from .tensor_ops import rsqrt_from_tensor
        return rsqrt_from_tensor(self)

    def sign(self) -> "Tensor":
        from .tensor_ops import sign_from_tensor
        return sign_from_tensor(self)

    def sgn(self) -> "Tensor":
        from .tensor_ops import sgn_from_tensor
        return sgn_from_tensor(self)

    def isnan(self) -> "Tensor":
        from .tensor_ops import isnan_from_tensor
        return isnan_from_tensor(self)

    def isinf(self) -> "Tensor":
        from .tensor_ops import isinf_from_tensor
        return isinf_from_tensor(self)

    def isfinite(self) -> "Tensor":
        from .tensor_ops import isfinite_from_tensor
        return isfinite_from_tensor(self)

    def isposinf(self) -> "Tensor":
        from .tensor_ops import isposinf_from_tensor
        return isposinf_from_tensor(self)

    def isneginf(self) -> "Tensor":
        from .tensor_ops import isneginf_from_tensor
        return isneginf_from_tensor(self)

    def logical_not(self) -> "Tensor":
        from .tensor_ops import logical_not_from_tensor
        return logical_not_from_tensor(self)

    def erf(self) -> "Tensor":
        from .tensor_ops import erf_from_tensor
        return erf_from_tensor(self)

    def erfc(self) -> "Tensor":
        from .tensor_ops import erfc_from_tensor
        return erfc_from_tensor(self)

    def lgamma(self) -> "Tensor":
        from .tensor_ops import lgamma_from_tensor
        return lgamma_from_tensor(self)

    def digamma(self) -> "Tensor":
        from .tensor_ops import digamma_from_tensor
        return digamma_from_tensor(self)

    def i0(self) -> "Tensor":
        from .tensor_ops import i0_from_tensor
        return i0_from_tensor(self)

    def deg2rad(self) -> "Tensor":
        from .tensor_ops import deg2rad_from_tensor
        return deg2rad_from_tensor(self)

    def rad2deg(self) -> "Tensor":
        from .tensor_ops import rad2deg_from_tensor
        return rad2deg_from_tensor(self)

    def eq(self, other: "Tensor") -> "Tensor":
        from .tensor_ops import eq_from_tensors
        return eq_from_tensors(self, other)

    def ne(self, other: "Tensor") -> "Tensor":
        from .tensor_ops import ne_from_tensors
        return ne_from_tensors(self, other)

    def gt(self, other: "Tensor | int | float") -> "Tensor":
        import torch as _torch
        if not isinstance(other, Tensor):
            other = _torch.tensor(other, dtype=self._dtype)
        from .tensor_ops import gt_from_tensors
        return gt_from_tensors(self, other)

    def lt(self, other: "Tensor | int | float") -> "Tensor":
        import torch as _torch
        if not isinstance(other, Tensor):
            other = _torch.tensor(other, dtype=self._dtype)
        from .tensor_ops import lt_from_tensors
        return lt_from_tensors(self, other)

    def ge(self, other: "Tensor | int | float") -> "Tensor":
        import torch as _torch
        if not isinstance(other, Tensor):
            other = _torch.tensor(other, dtype=self._dtype)
        from .tensor_ops import ge_from_tensors
        return ge_from_tensors(self, other)

    def le(self, other: "Tensor | int | float") -> "Tensor":
        import torch as _torch
        if not isinstance(other, Tensor):
            other = _torch.tensor(other, dtype=self._dtype)
        from .tensor_ops import le_from_tensors
        return le_from_tensors(self, other)

    def sum(self, dim: int | None = None, keepdim: bool = False) -> "Tensor":
        if dim is not None:
            from .tensor_ops import sum_dim_from_tensor
            return sum_dim_from_tensor(self, dim, keepdim)
        from .tensor_ops import sum_from_tensor
        return sum_from_tensor(self)

    def mean(self, dim: int | None = None, keepdim: bool = False) -> "Tensor":
        if dim is not None:
            from .tensor_ops import mean_dim_from_tensor
            return mean_dim_from_tensor(self, dim, keepdim)
        from .tensor_ops import mean_from_tensor
        return mean_from_tensor(self)

    def prod(self) -> "Tensor":
        from .tensor_ops import prod_from_tensor
        return prod_from_tensor(self)

    def min(self) -> "Tensor":
        from .tensor_ops import min_from_tensor
        return min_from_tensor(self)

    def max(self) -> "Tensor":
        from .tensor_ops import max_from_tensor
        return max_from_tensor(self)

    def masked_select(self, mask: "Tensor") -> "Tensor":
        from .tensor_ops import masked_select_from_tensor
        return masked_select_from_tensor(self, mask)

    def masked_fill(self, mask: "Tensor", value: float) -> "Tensor":
        from .tensor_ops import masked_fill_from_tensor
        return masked_fill_from_tensor(self, mask, value)

    def _set(self, other: "Tensor") -> None:
        self._id = other._id
        self._shape = list(other._shape)
        self._dtype = other._dtype

    def __getitem__(self, key: object) -> object:
        from ._tensor_indexing import getitem_from_tensor
        return getitem_from_tensor(self, key)


from .tensor_ops import (
    add_from_tensors,
    sub_from_tensors,
    mul_from_tensors,
    div_from_tensors,
    pow_from_tensors,
    heaviside_from_tensors,
    maximum_from_tensors,
    minimum_from_tensors,
    any_from_tensor,
    all_from_tensor,
    cumsum_from_tensor,
    cumprod_from_tensor,
    tril_from_tensor,
    triu_from_tensor,
    flip_from_tensor,
    where_from_tensors,
    topk_from_tensor,
    sort_from_tensor,
    gather_from_tensor,
    scatter_from_tensor,
    cat_from_tensors,
    stack_from_tensors,
    expand_from_tensor,
    index_select_from_tensor,
    sigmoid_from_tensor,
    tanh_from_tensor,
    sin_from_tensor,
    cos_from_tensor,
    gelu_from_tensor,
    silu_from_tensor,
    leaky_relu_from_tensor,
    floor_from_tensor,
    ceil_from_tensor,
    round_from_tensor,
    reciprocal_from_tensor,
    square_from_tensor,
    tan_from_tensor,
    asin_from_tensor,
    acos_from_tensor,
    atan_from_tensor,
    sinh_from_tensor,
    cosh_from_tensor,
    asinh_from_tensor,
    acosh_from_tensor,
    atanh_from_tensor,
    exp2_from_tensor,
    log2_from_tensor,
    log10_from_tensor,
    log1p_from_tensor,
    expm1_from_tensor,
    trunc_from_tensor,
    frac_from_tensor,
    softplus_from_tensor,
    mish_from_tensor,
    hardsigmoid_from_tensor,
    hardswish_from_tensor,
    softsign_from_tensor,
    tanhshrink_from_tensor,
    rsqrt_from_tensor,
    sign_from_tensor,
    sgn_from_tensor,
    isnan_from_tensor,
    isinf_from_tensor,
    isfinite_from_tensor,
    isposinf_from_tensor,
    isneginf_from_tensor,
    logical_not_from_tensor,
    erf_from_tensor,
    erfc_from_tensor,
    lgamma_from_tensor,
    digamma_from_tensor,
    i0_from_tensor,
    deg2rad_from_tensor,
    rad2deg_from_tensor,
    eq_from_tensors,
    ne_from_tensors,
    lt_from_tensors,
    le_from_tensors,
    gt_from_tensors,
    ge_from_tensors,
    sum_dim_from_tensor,
    sum_from_tensor,
    mean_from_tensor,
    mean_dim_from_tensor,
    prod_from_tensor,
    min_from_tensor,
    max_from_tensor,
    masked_select_from_tensor,
    masked_fill_from_tensor,
    softmax_from_tensor,
    log_softmax_from_tensor,
    relu_from_tensor,
    abs_from_tensor,
    sqrt_from_tensor,
    exp_from_tensor,
    log_from_tensor,
    neg_from_tensor,
    select_from_tensor,
    slice_from_tensor,
    matmul_from_tensors,
    _js_meta_to_tuple,
    _js_handle_array_to_tensors,
)

from .tensor_nn_ops import (
    conv2d_from_tensors,
    max_pool2d_from_tensor,
    avg_pool2d_from_tensor,
    batch_norm_from_tensor,
    nll_loss_from_tensor,
    cross_entropy_from_tensor,
    batch_norm_inference_from_tensor,
    layer_norm_from_tensor,
)

from .tensor_backward_ops import (
    conv2d_input_backward_from_tensors,
    conv2d_weight_backward_from_tensors,
    conv2d_bias_backward_from_tensors,
    softmax_backward_from_tensors,
    log_softmax_backward_from_tensors,
    nll_loss_backward_from_tensors,
    slice_backward_from_tensors,
    sort_backward_from_tensors,
    topk_backward_from_tensors,
    cross_entropy_backward_from_tensors,
    maxmin_backward_from_tensors,
)
