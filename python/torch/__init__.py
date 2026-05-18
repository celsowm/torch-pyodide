from __future__ import annotations

from typing import Sequence

from . import cuda
from ._tensor import (
    Tensor,
    arange_from_values,
    cat_from_tensors,
    expand_from_tensor,
    full_from_shape,
    full_like_from_tensor,
    zeros_like_from_tensor,
    ones_like_from_tensor,
    index_select_from_tensor,
    ones_from_shape,
    rand_from_shape,
    randn_from_shape,
    stack_from_tensors,
    tensor_from_data,
    where_from_tensors,
    zeros_from_shape,
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
    eq_from_tensors,
    ne_from_tensors,
    lt_from_tensors,
    le_from_tensors,
    gt_from_tensors,
    ge_from_tensors,
    sum_dim_from_tensor,
    mean_dim_from_tensor,
    prod_from_tensor,
    min_from_tensor,
    max_from_tensor,
    masked_select_from_tensor,
    masked_fill_from_tensor,
)

__all__ = [
    "Tensor",
    "cuda",
    "tensor",
    "zeros",
    "ones",
    "rand",
    "randn",
    "arange",
    "full",
    "full_like",
    "zeros_like",
    "ones_like",
    "add",
    "sub",
    "mul",
    "div",
    "matmul",
    "relu",
    "abs",
    "sqrt",
    "exp",
    "log",
    "neg",
    "clamp",
    "where",
    "argmax",
    "argmin",
    "reshape",
    "flatten",
    "squeeze",
    "unsqueeze",
    "transpose",
    "permute",
    "select",
    "slice",
    "cat",
    "stack",
    "expand",
    "index_select",
    "sigmoid",
    "tanh",
    "sin",
    "cos",
    "gelu",
    "silu",
    "leaky_relu",
    "floor",
    "ceil",
    "round",
    "reciprocal",
    "square",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "sum",
    "mean",
    "prod",
    "min",
    "max",
    "masked_select",
    "masked_fill",
]


def tensor(data: object, dtype: str = "float32") -> Tensor:
    return tensor_from_data(data, dtype=dtype)


def zeros(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    return zeros_from_shape(shape, dtype=dtype)


def ones(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    return ones_from_shape(shape, dtype=dtype)


def rand(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    return rand_from_shape(shape, dtype=dtype)


def randn(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    return randn_from_shape(shape, dtype=dtype)


def arange(
    start: float,
    end: float | None = None,
    step: float = 1.0,
    dtype: str = "float32",
) -> Tensor:
    return arange_from_values(start=start, end=end, step=step, dtype=dtype)


def full(shape: int | Sequence[int], fill_value: float, dtype: str = "float32") -> Tensor:
    return full_from_shape(shape=shape, fill_value=fill_value, dtype=dtype)


def full_like(input: Tensor, fill_value: float, dtype: str | None = None) -> Tensor:
    return full_like_from_tensor(input, fill_value=fill_value, dtype=dtype)


def zeros_like(input: Tensor, dtype: str | None = None) -> Tensor:
    return zeros_like_from_tensor(input, dtype=dtype)


def ones_like(input: Tensor, dtype: str | None = None) -> Tensor:
    return ones_like_from_tensor(input, dtype=dtype)


def add(a: Tensor, b: Tensor) -> Tensor:
    return a.add(b)


def sub(a: Tensor, b: Tensor) -> Tensor:
    return a.sub(b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    return a.mul(b)


def div(a: Tensor, b: Tensor) -> Tensor:
    return a.div(b)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a.matmul(b)


def relu(x: Tensor) -> Tensor:
    return x.relu()


def abs(x: Tensor) -> Tensor:
    return x.abs()


def sqrt(x: Tensor) -> Tensor:
    return x.sqrt()


def exp(x: Tensor) -> Tensor:
    return x.exp()


def log(x: Tensor) -> Tensor:
    return x.log()


def neg(x: Tensor) -> Tensor:
    return x.neg()


def clamp(x: Tensor, min: float, max: float) -> Tensor:
    return x.clamp(min=min, max=max)


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return where_from_tensors(condition, x, y)


def argmax(x: Tensor) -> Tensor:
    return x.argmax()


def argmin(x: Tensor) -> Tensor:
    return x.argmin()


def reshape(input: Tensor, shape: int | Sequence[int]) -> Tensor:
    return input.reshape(shape)


def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    return input.flatten(start_dim=start_dim, end_dim=end_dim)


def squeeze(input: Tensor, dim: int | None = None) -> Tensor:
    return input.squeeze(dim=dim)


def unsqueeze(input: Tensor, dim: int) -> Tensor:
    return input.unsqueeze(dim=dim)


def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    return input.transpose(dim0=dim0, dim1=dim1)


def permute(input: Tensor, dims: Sequence[int]) -> Tensor:
    return input.permute(dims=dims)


def select(input: Tensor, dim: int, index: int) -> Tensor:
    return input.select(dim=dim, index=index)


def slice(input: Tensor, dim: int, start: int | None = None, end: int | None = None, step: int = 1) -> Tensor:
    return input.slice(dim=dim, start=start, end=end, step=step)


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    return cat_from_tensors(tensors, dim=dim)


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    return stack_from_tensors(tensors, dim=dim)


def expand(input: Tensor, shape: int | Sequence[int]) -> Tensor:
    return expand_from_tensor(input, shape=shape)


def index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
    return index_select_from_tensor(input, dim=dim, index=index)


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def sin(x: Tensor) -> Tensor:
    return x.sin()


def cos(x: Tensor) -> Tensor:
    return x.cos()


def gelu(x: Tensor) -> Tensor:
    return x.gelu()


def silu(x: Tensor) -> Tensor:
    return x.silu()


def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    return x.leaky_relu(alpha)


def floor(x: Tensor) -> Tensor:
    return x.floor()


def ceil(x: Tensor) -> Tensor:
    return x.ceil()


def round(x: Tensor) -> Tensor:
    return x.round()


def reciprocal(x: Tensor) -> Tensor:
    return x.reciprocal()


def square(x: Tensor) -> Tensor:
    return x.square()


def eq(a: Tensor, b: Tensor) -> Tensor:
    return a.eq(b)


def ne(a: Tensor, b: Tensor) -> Tensor:
    return a.ne(b)


def lt(a: Tensor, b: Tensor) -> Tensor:
    return a.lt(b)


def le(a: Tensor, b: Tensor) -> Tensor:
    return a.le(b)


def gt(a: Tensor, b: Tensor) -> Tensor:
    return a.gt(b)


def ge(a: Tensor, b: Tensor) -> Tensor:
    return a.ge(b)


def sum(input: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    return input.sum(dim=dim, keepdim=keepdim)


def mean(input: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    return input.mean(dim=dim, keepdim=keepdim)


def prod(input: Tensor) -> Tensor:
    return input.prod()


def min(input: Tensor) -> Tensor:
    return input.min()


def max(input: Tensor) -> Tensor:
    return input.max()


def masked_select(input: Tensor, mask: Tensor) -> Tensor:
    return masked_select_from_tensor(input, mask=mask)


def masked_fill(input: Tensor, mask: Tensor, value: float) -> Tensor:
    return masked_fill_from_tensor(input, mask=mask, value=value)
