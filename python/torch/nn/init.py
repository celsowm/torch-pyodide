from __future__ import annotations

import math
from typing import Sequence


def _calculate_fan_in_and_fan_out(tensor: object) -> tuple[int, int]:
    ndim = tensor.ndim  # type: ignore[union-attr]
    if ndim < 2:
        raise ValueError(f"tensor must have at least 2 dims, got {ndim}")
    n_dims = len(tensor.shape)  # type: ignore[union-attr]
    receptive_field_size = 1
    for i in range(2, n_dims):
        receptive_field_size *= tensor.shape[i]  # type: ignore[union-attr]
    fan_in = tensor.shape[1] * receptive_field_size  # type: ignore[union-attr]
    fan_out = tensor.shape[0] * receptive_field_size  # type: ignore[union-attr]
    return fan_in, fan_out


def uniform_(tensor: object, a: float = 0.0, b: float = 1.0) -> object:
    from torch import Tensor
    span = b - a
    r = _rand_gpu(list(tensor.shape), tensor.dtype)
    result = r.mul(span).add(a)
    tensor._set(result)
    return tensor


def normal_(tensor: object, mean: float = 0.0, std: float = 1.0) -> object:
    from torch import Tensor
    r = _randn_gpu(list(tensor.shape), tensor.dtype)
    result = r.mul(std).add(mean)
    tensor._set(result)
    return tensor


def constant_(tensor: object, val: float) -> object:
    from torch import Tensor
    import torch as _torch
    result = _torch.full(list(tensor.shape), val, dtype=tensor.dtype)
    tensor._set(result)
    return tensor


def zeros_(tensor: object) -> object:
    import torch as _torch
    z = _torch.zeros(list(tensor.shape), dtype=tensor.dtype)
    tensor._set(z)
    return tensor


def ones_(tensor: object) -> object:
    import torch as _torch
    o = _torch.ones(list(tensor.shape), dtype=tensor.dtype)
    tensor._set(o)
    return tensor


def xavier_uniform_(tensor: object, gain: float = 1.0) -> object:
    import torch as _torch
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    r = _rand_gpu(list(tensor.shape), tensor.dtype)
    result = r.mul(2 * bound).sub(bound)
    tensor._set(result)
    return tensor


def xavier_normal_(tensor: object, gain: float = 1.0) -> object:
    import torch as _torch
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    r = _randn_gpu(list(tensor.shape), tensor.dtype)
    result = r.mul(std)
    tensor._set(result)
    return tensor


def kaiming_uniform_(tensor: object, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> object:
    import torch as _torch
    fan, _ = _calculate_fan_in_and_fan_out(tensor)
    gain = math.sqrt(2.0 / (1 + a ** 2)) if nonlinearity == "leaky_relu" else 1.0
    bound = gain * math.sqrt(3.0 / max(fan, 1))
    r = _rand_gpu(list(tensor.shape), tensor.dtype)
    result = r.mul(2 * bound).sub(bound)
    tensor._set(result)
    return tensor


def kaiming_normal_(tensor: object, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> object:
    import torch as _torch
    fan, _ = _calculate_fan_in_and_fan_out(tensor)
    gain = math.sqrt(2.0 / (1 + a ** 2)) if nonlinearity == "leaky_relu" else 1.0
    std = gain / math.sqrt(max(fan, 1))
    r = _randn_gpu(list(tensor.shape), tensor.dtype)
    result = r.mul(std)
    tensor._set(result)
    return tensor


def orthogonal_(tensor: object, gain: float = 1.0) -> object:
    import torch as _torch
    if tensor.ndim < 2:
        raise ValueError("orthogonal_ requires at least 2 dimensions")
    rows, cols = tensor.shape[0], tensor.shape[1]
    raw = _randn_gpu([rows, cols], tensor.dtype)
    flat = raw.flatten().tolist()
    n = len(flat)
    # Gram-Schmidt orthogonalization (CPU, unavoidable without QR GPU)
    q_rows: list[list[float]] = []
    for i in range(rows):
        v = flat[i * cols:(i + 1) * cols]
        for j in range(len(q_rows)):
            q = q_rows[j]
            dot_vq = sum(a * b for a, b in zip(v, q))
            dot_qq = sum(a * a for a in q)
            if dot_qq > 1e-10:
                factor = dot_vq / dot_qq
                v = [a - factor * b for a, b in zip(v, q)]
        norm_v = math.sqrt(sum(a * a for a in v))
        if norm_v > 1e-10:
            v = [a / norm_v for a in v]
        q_rows.append(v)
    out: list[float] = []
    for row in q_rows:
        out.extend(row)
    while len(out) < n:
        out.append(0.0)
    out = [gain * v for v in out]
    _assign_flat(tensor, out)
    return tensor


def dirac_(tensor: object, groups: int = 1) -> object:
    import torch as _torch
    if tensor.ndim not in (3, 4, 5):
        raise ValueError("dirac_ only supports 3D, 4D, or 5D tensors")
    result = _torch.zeros(list(tensor.shape), dtype=tensor.dtype)
    dim1, dim2 = tensor.shape[0], tensor.shape[1]
    if dim1 % groups != 0:
        raise ValueError("tensor size must be divisible by groups")
    dim2_per_group = dim2 // groups
    min_dim = min(dim1, dim2_per_group)
    for d in range(min_dim):
        nd = tensor.ndim
        if nd == 3:
            kh_kw = tensor.shape[2]
            center = kh_kw // 2
            result[d, d * groups + d // dim1, center] = 1.0
        elif nd == 4:
            kh, kw = tensor.shape[2], tensor.shape[3]
            result[d, d * groups + d // dim1, kh // 2, kw // 2] = 1.0
        elif nd == 5:
            kh, kw, kd = tensor.shape[2], tensor.shape[3], tensor.shape[4]
            result[d, d * groups + d // dim1, kh // 2, kw // 2, kd // 2] = 1.0
    tensor._set(result)
    return tensor


def eye_(tensor: object) -> object:
    import torch as _torch
    if tensor.ndim != 2:
        raise ValueError("eye_ only supports 2D tensors")
    rows, cols = tensor.shape[0], tensor.shape[1]
    result = _torch.zeros([rows, cols], dtype=tensor.dtype)
    min_dim = min(rows, cols)
    for i in range(min_dim):
        result[i, i] = 1.0
    tensor._set(result)
    return tensor


def _rand_gpu(shape: list[int], dtype: str) -> object:
    import torch as _torch
    return _torch.rand(shape, dtype=dtype)


def _randn_gpu(shape: list[int], dtype: str) -> object:
    import torch as _torch
    return _torch.randn(shape, dtype=dtype)


def _assign_flat(tensor: object, flat: list[float]) -> object:
    from torch._tensor import _reshape_flat_values, tensor_from_data
    reshaped = _reshape_flat_values(flat, list(tensor.shape))
    new_tensor = tensor_from_data(reshaped, tensor.dtype)
    tensor._set(new_tensor)
    return tensor
