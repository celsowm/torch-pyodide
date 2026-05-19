from __future__ import annotations

import math
import time
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


def _randn_like_flat(n: int) -> list[float]:
    """Generate n random values from standard normal using Box-Muller."""
    import random
    random.seed(int(time.time() * 1000) + n)
    out: list[float] = []
    for _ in range(0, n, 2):
        u1 = random.random()
        u2 = random.random()
        if u1 < 1e-10:
            u1 = 1e-10
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        out.append(z0)
        if len(out) < n:
            out.append(z1)
    return out[:n]


def _rand_flat(n: int, a: float, b: float) -> list[float]:
    """Generate n random values uniformly in [a, b]."""
    import random
    random.seed(int(time.time() * 1000) + n)
    return [a + (b - a) * random.random() for _ in range(n)]


def uniform_(tensor: object, a: float = 0.0, b: float = 1.0) -> object:
    """Fills the tensor with values drawn from U(a, b)."""
    from torch import Tensor
    data = tensor.tolist()  # type: ignore[union-attr]
    flat = _flatten(data)
    n = len(flat)
    out = _rand_flat(n, a, b)
    return _assign_flat(tensor, out)


def normal_(tensor: object, mean: float = 0.0, std: float = 1.0) -> object:
    """Fills the tensor with values drawn from N(mean, std)."""
    from torch import Tensor
    data = tensor.tolist()  # type: ignore[union-attr]
    flat = _flatten(data)
    n = len(flat)
    raw = _randn_like_flat(n)
    out = [mean + std * v for v in raw]
    return _assign_flat(tensor, out)


def constant_(tensor: object, val: float) -> object:
    """Fills the tensor with a constant value."""
    from torch import Tensor
    data = tensor.tolist()  # type: ignore[union-attr]
    flat = _flatten(data)
    n = len(flat)
    return _assign_flat(tensor, [val] * n)


def zeros_(tensor: object) -> object:
    """Fills the tensor with zeros."""
    return constant_(tensor, 0.0)


def ones_(tensor: object) -> object:
    """Fills the tensor with ones."""
    return constant_(tensor, 1.0)


def xavier_uniform_(tensor: object, gain: float = 1.0) -> object:
    """Fills with values from U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))."""
    from torch import Tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    data = tensor.tolist()  # type: ignore[union-attr]
    flat = _flatten(data)
    n = len(flat)
    out = _rand_flat(n, -bound, bound)
    return _assign_flat(tensor, out)


def xavier_normal_(tensor: object, gain: float = 1.0) -> object:
    """Fills with values from N(0, std) where std = gain * sqrt(2 / (fan_in + fan_out))."""
    from torch import Tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    data = tensor.tolist()  # type: ignore[union-attr]
    flat = _flatten(data)
    n = len(flat)
    raw = _randn_like_flat(n)
    out = [std * v for v in raw]
    return _assign_flat(tensor, out)


def kaiming_uniform_(tensor: object, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> object:
    """Fills with values from U(-bound, bound) using He initialization."""
    from torch import Tensor
    fan, _ = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        gain = math.sqrt(2.0 / (1 + a ** 2)) if nonlinearity == "leaky_relu" else 1.0
        bound = gain * math.sqrt(3.0 / max(fan, 1))
    else:
        gain = math.sqrt(2.0 / (1 + a ** 2)) if nonlinearity == "leaky_relu" else 1.0
        bound = gain * math.sqrt(3.0 / max(fan, 1))
    data = tensor.tolist()  # type: ignore[union-attr]
    flat = _flatten(data)
    n = len(flat)
    out = _rand_flat(n, -bound, bound)
    return _assign_flat(tensor, out)


def kaiming_normal_(tensor: object, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> object:
    """Fills with values from N(0, std) using He initialization."""
    from torch import Tensor
    fan, _ = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        gain = math.sqrt(2.0 / (1 + a ** 2)) if nonlinearity == "leaky_relu" else 1.0
        std = gain / math.sqrt(max(fan, 1))
    else:
        gain = math.sqrt(2.0 / (1 + a ** 2)) if nonlinearity == "leaky_relu" else 1.0
        std = gain / math.sqrt(max(fan, 1))
    data = tensor.tolist()  # type: ignore[union-attr]
    flat = _flatten(data)
    n = len(flat)
    raw = _randn_like_flat(n)
    out = [std * v for v in raw]
    return _assign_flat(tensor, out)


def orthogonal_(tensor: object, gain: float = 1.0) -> object:
    """Fills with a (semi) orthogonal matrix."""
    from torch import Tensor
    if tensor.ndim < 2:
        raise ValueError("orthogonal_ requires at least 2 dimensions")
    rows = tensor.shape[0]
    cols = tensor.shape[1]
    flat = _flatten(tensor.tolist())  # type: ignore[union-attr]
    n = rows * cols
    raw = _randn_like_flat(n)

    # Build matrix and do QR decomposition (simplified 2D)
    matrix = []
    idx = 0
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(raw[idx])
            idx += 1
        matrix.append(row)

    # Gram-Schmidt orthogonalization
    q_rows: list[list[float]] = []
    for i in range(rows):
        v = matrix[i][:]
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

    # Flatten result
    out: list[float] = []
    for row in q_rows:
        out.extend(row)
    # Pad if needed
    while len(out) < n:
        out.append(0.0)
    out = [gain * v for v in out]
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
