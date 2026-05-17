from __future__ import annotations

from typing import Sequence

from . import cuda
from ._tensor import (
    Tensor,
    ones_from_shape,
    rand_from_shape,
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
    "add",
    "sub",
    "mul",
    "div",
    "matmul",
    "relu",
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


def clamp(x: Tensor, min: float, max: float) -> Tensor:
    return x.clamp(min=min, max=max)


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return where_from_tensors(condition, x, y)


def argmax(x: Tensor) -> Tensor:
    return x.argmax()


def argmin(x: Tensor) -> Tensor:
    return x.argmin()
