from __future__ import annotations

from typing import Sequence

from ._runtime import _get_runtime
from ._tensor import Tensor
from .tensor_factories_ops import (
    arange_from_values,
    empty_from_shape,
    empty_like_from_tensor,
    full_from_shape,
    full_like_from_tensor,
    ones_from_shape,
    ones_like_from_tensor,
    rand_from_shape,
    randn_from_shape,
    tensor_from_data,
    zeros_from_shape,
    zeros_like_from_tensor,
)


def tensor(data: object, dtype: str = "float32", requires_grad: bool = False) -> Tensor:
    return tensor_from_data(data, dtype=dtype, requires_grad=requires_grad)


def zeros(shape: int | Sequence[int], dtype: str = "float32", *, requires_grad: bool = False) -> Tensor:
    result = zeros_from_shape(shape, dtype=dtype)
    if requires_grad:
        result.requires_grad_()
    return result


def ones(shape: int | Sequence[int], dtype: str = "float32", *, requires_grad: bool = False) -> Tensor:
    result = ones_from_shape(shape, dtype=dtype)
    if requires_grad:
        result.requires_grad_()
    return result


def rand(shape: int | Sequence[int], dtype: str = "float32", *, requires_grad: bool = False) -> Tensor:
    result = rand_from_shape(shape, dtype=dtype)
    if requires_grad:
        result.requires_grad_()
    return result


def randn(shape: int | Sequence[int], dtype: str = "float32", *, requires_grad: bool = False) -> Tensor:
    result = randn_from_shape(shape, dtype=dtype)
    if requires_grad:
        result.requires_grad_()
    return result


def manual_seed(seed: int) -> None:
    _get_runtime().setSeed(seed)


def seed() -> int:
    import random

    s = random.randint(0, 2**31 - 1)
    manual_seed(s)
    return s


def arange(
    start: float,
    end: float | None = None,
    step: float = 1.0,
    dtype: str = "float32",
) -> Tensor:
    return arange_from_values(start=start, end=end, step=step, dtype=dtype)


def full(shape: int | Sequence[int], fill_value: float, dtype: str = "float32", *, requires_grad: bool = False) -> Tensor:
    result = full_from_shape(shape=shape, fill_value=fill_value, dtype=dtype)
    if requires_grad:
        result.requires_grad_()
    return result


def full_like(input: Tensor, fill_value: float, dtype: str | None = None) -> Tensor:
    return full_like_from_tensor(input, fill_value=fill_value, dtype=dtype)


def zeros_like(input: Tensor, dtype: str | None = None) -> Tensor:
    return zeros_like_from_tensor(input, dtype=dtype)


def ones_like(input: Tensor, dtype: str | None = None) -> Tensor:
    return ones_like_from_tensor(input, dtype=dtype)


def empty(shape: int | Sequence[int], dtype: str = "float32", *, requires_grad: bool = False) -> Tensor:
    result = empty_from_shape(shape, dtype=dtype)
    if requires_grad:
        result.requires_grad_()
    return result


def empty_like(input: Tensor, dtype: str | None = None) -> Tensor:
    return empty_like_from_tensor(input, dtype=dtype)


def eye(n: int, m: int | None = None, dtype: str = "float32") -> Tensor:
    rows = n
    cols = m if m is not None else n
    result = zeros([rows, cols], dtype=dtype)
    min_dim = min(rows, cols)
    for i in range(min_dim):
        result[i, i] = 1.0
    return result


def randint(low: int, high: int | None = None, size: int | Sequence[int] | None = None, dtype: str = "int64") -> Tensor:
    if high is None:
        low, high = 0, low
    if size is None:
        size = [1]
    if isinstance(size, int):
        size = [size]
    r = rand(list(size))
    span = float(high - low)
    scaled = r.mul(span).add(float(low))
    return scaled.to(dtype)


def randperm(n: int, dtype: str = "int64") -> Tensor:
    r = rand([n])
    _, indices = r.sort(dim=0)
    return indices.to(dtype)


def linspace(start: float, end: float, steps: int, dtype: str = "float32") -> Tensor:
    if steps < 2:
        return full([steps], start, dtype=dtype)
    step = (end - start) / (steps - 1)
    return arange(start=start, end=end + step * 0.5, step=step, dtype=dtype)


def logspace(start: float, end: float, steps: int, dtype: str = "float32") -> Tensor:
    return linspace(start, end, steps, dtype=dtype).pow(10.0)
