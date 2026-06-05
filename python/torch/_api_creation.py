from __future__ import annotations

from typing import Sequence
import random as _random

from ._runtime import _get_runtime
from ._tensor import Tensor
from .tensor_shape_utils import _normalize_shape_from_args
from .tensor_factories_ops import (
    arange_from_values,
    bernoulli_from_shape,
    empty_from_shape,
    empty_like_from_tensor,
    exponential_from_shape,
    full_from_shape,
    full_like_from_tensor,
    log_normal_from_shape,
    normal_from_shape,
    ones_from_shape,
    ones_like_from_tensor,
    rand_from_shape,
    randn_from_shape,
    tensor_from_data,
    zeros_from_shape,
    zeros_like_from_tensor,
)


def _normalize_factory_shape_args(size: tuple[object, ...], dtype: str) -> tuple[list[int], str]:
    if not size:
        raise TypeError("missing required size argument")
    if isinstance(size[-1], str):
        if dtype != "float32":
            raise TypeError("dtype specified both positionally and by keyword")
        dtype = size[-1]
        size = size[:-1]
        if not size:
            raise TypeError("missing required size argument")
    return _normalize_shape_from_args(size), dtype


def _normalize_device(device: object = None) -> None:
    if device is None:
        return
    if str(device).lower() == "cpu":
        return
    raise RuntimeError(f"Only CPU device is supported, received: {device!r}.")


def tensor(data: object, dtype: str = "float32", requires_grad: bool = False, device: object = None) -> Tensor:
    _normalize_device(device)
    return tensor_from_data(data, dtype=dtype, requires_grad=requires_grad)


def zeros(*size: object, dtype: str = "float32", device: object = None, requires_grad: bool = False) -> Tensor:
    _normalize_device(device)
    normalized_shape, dtype = _normalize_factory_shape_args(size, dtype)
    result = zeros_from_shape(normalized_shape, dtype=dtype)
    if requires_grad:
        result.requires_grad_()
    return result


def ones(*size: object, dtype: str = "float32", device: object = None, requires_grad: bool = False) -> Tensor:
    _normalize_device(device)
    normalized_shape, dtype = _normalize_factory_shape_args(size, dtype)
    result = ones_from_shape(normalized_shape, dtype=dtype)
    if requires_grad:
        result.requires_grad_()
    return result


def rand(*size: object, dtype: str = "float32", device: object = None, requires_grad: bool = False) -> Tensor:
    _normalize_device(device)
    normalized_shape, dtype = _normalize_factory_shape_args(size, dtype)
    result = rand_from_shape(normalized_shape, dtype=dtype)
    if requires_grad:
        result.requires_grad_()
    return result


def randn(*size: object, dtype: str = "float32", device: object = None, requires_grad: bool = False) -> Tensor:
    _normalize_device(device)
    normalized_shape, dtype = _normalize_factory_shape_args(size, dtype)
    result = randn_from_shape(normalized_shape, dtype=dtype)
    if requires_grad:
        result.requires_grad_()
    return result


def manual_seed(seed: int) -> None:
    _random.seed(int(seed))
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
    device: object = None,
) -> Tensor:
    _normalize_device(device)
    return arange_from_values(start=start, end=end, step=step, dtype=dtype)


def full(
    size: int | Sequence[int],
    fill_value: float,
    dtype: str = "float32",
    *,
    device: object = None,
    requires_grad: bool = False,
) -> Tensor:
    _normalize_device(device)
    result = full_from_shape(shape=size, fill_value=fill_value, dtype=dtype)
    if requires_grad:
        result.requires_grad_()
    return result


def full_like(input: Tensor, fill_value: float, dtype: str | None = None, device: object = None) -> Tensor:
    _normalize_device(device)
    return full_like_from_tensor(input, fill_value=fill_value, dtype=dtype)


def zeros_like(input: Tensor, dtype: str | None = None, device: object = None) -> Tensor:
    _normalize_device(device)
    return zeros_like_from_tensor(input, dtype=dtype)


def ones_like(input: Tensor, dtype: str | None = None, device: object = None) -> Tensor:
    _normalize_device(device)
    return ones_like_from_tensor(input, dtype=dtype)


def empty(*size: object, dtype: str = "float32", device: object = None, requires_grad: bool = False) -> Tensor:
    _normalize_device(device)
    normalized_shape, dtype = _normalize_factory_shape_args(size, dtype)
    result = empty_from_shape(normalized_shape, dtype=dtype)
    if requires_grad:
        result.requires_grad_()
    return result


def empty_like(input: Tensor, dtype: str | None = None, device: object = None) -> Tensor:
    _normalize_device(device)
    return empty_like_from_tensor(input, dtype=dtype)


def eye(n: int, m: int | None = None, dtype: str = "float32", device: object = None) -> Tensor:
    _normalize_device(device)
    rows = n
    cols = m if m is not None else n
    result = zeros([rows, cols], dtype=dtype)
    min_dim = min(rows, cols)
    for i in range(min_dim):
        result[i, i] = 1.0
    return result


def randint(
    low: int,
    high: int | None = None,
    size: int | Sequence[int] | None = None,
    dtype: str = "int64",
    device: object = None,
) -> Tensor:
    _normalize_device(device)
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


def randperm(n: int, dtype: str = "int64", device: object = None) -> Tensor:
    _normalize_device(device)
    r = rand([n])
    _, indices = r.sort(dim=0)
    return indices.to(dtype)


def _sample_multinomial_row(weights: Sequence[float], num_samples: int, replacement: bool) -> list[int]:
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if not replacement and num_samples > len(weights):
        raise ValueError("cannot sample n_sample > prob_dist.size(-1) samples without replacement")

    available = [(i, max(float(w), 0.0)) for i, w in enumerate(weights)]
    result: list[int] = []
    for _ in range(num_samples):
        total = sum(weight for _, weight in available)
        if total <= 0.0:
            raise RuntimeError("invalid multinomial distribution (sum of probabilities <= 0)")
        threshold = _random.random() * total
        cumulative = 0.0
        chosen_pos = len(available) - 1
        for pos, (_, weight) in enumerate(available):
            cumulative += weight
            if threshold <= cumulative:
                chosen_pos = pos
                break
        result.append(available[chosen_pos][0])
        if not replacement:
            available.pop(chosen_pos)
    return result


def multinomial(
    input: Tensor,
    num_samples: int,
    replacement: bool = False,
    *,
    generator: object = None,
) -> Tensor:
    if generator is not None:
        raise NotImplementedError("multinomial(generator=...) is not supported")
    if len(input.shape) not in (1, 2):
        raise RuntimeError("prob_dist must be 1 or 2 dim")

    values = input.tolist()
    if len(input.shape) == 1:
        return tensor(_sample_multinomial_row(values, int(num_samples), replacement), dtype="int64")

    rows = [_sample_multinomial_row(row, int(num_samples), replacement) for row in values]
    return tensor(rows, dtype="int64")


def linspace(start: float, end: float, steps: int, dtype: str = "float32", device: object = None) -> Tensor:
    _normalize_device(device)
    if steps < 2:
        return full([steps], start, dtype=dtype)
    step = (end - start) / (steps - 1)
    return arange(start=start, end=end + step * 0.5, step=step, dtype=dtype)


def logspace(start: float, end: float, steps: int, dtype: str = "float32", device: object = None) -> Tensor:
    _normalize_device(device)
    return linspace(start, end, steps, dtype=dtype).pow(10.0)


def normal(
    mean: float | Tensor,
    std: float | Tensor,
    *,
    size: int | Sequence[int] | None = None,
    dtype: str = "float32",
    device: object = None,
) -> Tensor:
    """torch.normal(mean, std, size=...) compatibility wrapper.

    When given two scalars, uses size; when given a Tensor mean/std, draws one
    sample per element of the broadcasted shape (PyTorch behavior).
    """
    _normalize_device(device)
    from ._tensor import Tensor

    if isinstance(mean, Tensor) or isinstance(std, Tensor):
        if not isinstance(mean, Tensor) or not isinstance(std, Tensor):
            raise TypeError("mean and std must both be Tensor or both be numbers")
        return _normal_tensor(mean, std, dtype=dtype)

    if size is None:
        raise TypeError("size is required when mean and std are both numbers")
    result = normal_from_shape(shape=size, mean=float(mean), std=float(std), dtype=dtype)
    if result.requires_grad is False:
        pass
    return result


def _normal_tensor(mean: "Tensor", std: "Tensor", dtype: str = "float32") -> "Tensor":
    """Sample a normal with per-element mean/std (PyTorch broadcast behavior)."""
    from ._runtime import _get_runtime, _run_js_awaitable
    from .tensor_factories_ops import _mk
    runtime = _get_runtime()
    out = randn_from_shape(list(mean.shape), dtype=dtype)
    std_num = std.tolist() if isinstance(std, Tensor) else float(std)
    mean_num = mean.tolist() if isinstance(mean, Tensor) else float(mean)
    # Decompose: out * std + mean. PyTorch treats std as broadcastable.
    return out.mul(std).add(mean)


def bernoulli(
    input: Tensor | float = 0.5,
    *,
    size: int | Sequence[int] | None = None,
    dtype: str = "float32",
    device: object = None,
) -> Tensor:
    """torch.bernoulli(input, size=...) compatibility wrapper."""
    _normalize_device(device)
    from ._tensor import Tensor
    if isinstance(input, Tensor):
        # Per-element Bernoulli from a probability tensor.
        probs = input.tolist()
        # Flatten arbitrarily nested lists.
        def _flatten_probs(values):
            if isinstance(values, (list, tuple)):
                out = []
                for v in values:
                    out.extend(_flatten_probs(v))
                return out
            return [float(values)]
        flat_probs = _flatten_probs(probs)
        flat = [1.0 if (p > 0) and ((p >= 1.0) or (_random.random() < p)) else 0.0 for p in flat_probs]
        from .tensor_factories_ops import tensor_from_data
        return tensor_from_data(flat, shape=list(input.shape), dtype=dtype)
    p = float(input)
    if size is None:
        size = [1]
    return bernoulli_from_shape(shape=size, p=p, dtype=dtype)


def exponential(
    lambd: float = 1.0,
    *,
    size: int | Sequence[int] | None = None,
    dtype: str = "float32",
    device: object = None,
) -> Tensor:
    """torch.exponential(lambd, size=...) compatibility wrapper."""
    _normalize_device(device)
    if size is None:
        size = [1]
    return exponential_from_shape(shape=size, lambd=lambd, dtype=dtype)


def log_normal(
    mean: float = 0.0,
    std: float = 1.0,
    size: int | Sequence[int] | None = None,
    dtype: str = "float32",
    device: object = None,
) -> Tensor:
    """torch.log_normal(mean, std, size=...) compatibility wrapper."""
    _normalize_device(device)
    if size is None:
        size = [1]
    return log_normal_from_shape(shape=size, mean=mean, std=std, dtype=dtype)
