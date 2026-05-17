from __future__ import annotations

from typing import Sequence

from ._runtime import _get_runtime, _run_js_awaitable
from ._tensor import Tensor, ones_from_shape, tensor_from_data, zeros_from_shape

__all__ = ["Tensor", "init", "tensor", "zeros", "ones", "add", "sub", "mul", "div", "matmul", "relu"]


def init() -> None:
    runtime = _get_runtime()
    _run_js_awaitable(runtime.init())


def tensor(data: object, dtype: str = "float32") -> Tensor:
    return tensor_from_data(data, dtype=dtype)


def zeros(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    return zeros_from_shape(shape, dtype=dtype)


def ones(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    return ones_from_shape(shape, dtype=dtype)


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
