from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    ndim = tensor.ndim
    if ndim < 2:
        raise ValueError(f"tensor must have at least 2 dims, got {ndim}")
    n_dims = len(tensor.shape)
    receptive_field_size = 1
    for i in range(2, n_dims):
        receptive_field_size *= tensor.shape[i]
    fan_in = tensor.shape[1] * receptive_field_size
    fan_out = tensor.shape[0] * receptive_field_size
    return fan_in, fan_out


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    data = tensor.tolist()
    flat = _flatten(data)
    n = len(flat)
    out = [a + (b - a) * (hash(str(i)) % 10000) / 10000.0 for i in range(n)]
    return _assign_flat(tensor, out)


def kaiming_uniform_(tensor: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Tensor:
    fan, _ = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        gain = math.sqrt(2.0 / (1 + a ** 2)) if nonlinearity == "leaky_relu" else 1.0
        bound = gain * math.sqrt(3.0 / max(fan, 1))
    else:
        bound = math.sqrt(3.0 / max(fan, 1))
    data = tensor.tolist()
    flat = _flatten(data)
    n = len(flat)
    out = [-bound + 2 * bound * (hash(str(i * 42)) % 10000) / 10000.0 for i in range(n)]
    return _assign_flat(tensor, out)


def _flatten(data: object) -> list[float]:
    if isinstance(data, list):
        out: list[float] = []
        for item in data:
            out.extend(_flatten(item))
        return out
    return [float(data)]


def _assign_flat(tensor: Tensor, flat: list[float]) -> Tensor:
    from torch._tensor import _reshape_flat_values, tensor_from_data
    reshaped = _reshape_flat_values(flat, list(tensor.shape))
    new_tensor = tensor_from_data(reshaped, tensor.dtype)
    tensor._set(new_tensor)
    return tensor
