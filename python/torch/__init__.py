from __future__ import annotations

from typing import Sequence

from . import cuda
from ._tensor import (
    Tensor,
    arange_from_values,
    full_from_shape,
    full_like_from_tensor,
    ones_from_shape,
    rand_from_shape,
    randn_from_shape,
    tensor_from_data,
    where_from_tensors,
    zeros_from_shape,
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
