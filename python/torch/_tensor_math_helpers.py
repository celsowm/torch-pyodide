from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._tensor import Tensor


def repeat_interleave_from_tensor(tensor: "Tensor", repeats: int, dim: int | None = None) -> "Tensor":
    from .__init__ import arange

    if dim is None:
        flat = tensor.flatten()
        shape_0 = flat._shape[0]
        indices = arange(shape_0, dtype="int64").unsqueeze(1).expand(shape_0, repeats).flatten()
        return flat.index_select(0, indices)
    d = dim if dim >= 0 else dim + tensor._shape.__len__()
    shape_d = tensor._shape[d]
    indices = arange(shape_d, dtype="int64").unsqueeze(1).expand(shape_d, repeats).flatten()
    return tensor.index_select(d, indices)


def norm_from_tensor(tensor: "Tensor", p: float | str = "fro") -> "Tensor":
    if p == "fro" or p == 2:
        return (tensor * tensor).sum().sqrt()
    if p == 1:
        return tensor.abs().sum()
    if p == float("inf") or p == "inf":
        return tensor.abs().max()
    return (tensor.abs() ** p).sum() ** (1.0 / p)


def radd_from_tensor(tensor: "Tensor", other: "Tensor | float") -> "Tensor":
    from .tensor_ops import _scalar_to_tensor

    return tensor.add(other) if isinstance(other, Tensor) else _scalar_to_tensor(float(other), tensor._dtype).add(tensor)


def rsub_from_tensor(tensor: "Tensor", other: "Tensor | float") -> "Tensor":
    from .tensor_ops import _scalar_to_tensor

    return tensor.neg().add(other) if isinstance(other, tensor.__class__) else _scalar_to_tensor(float(other), tensor._dtype).sub(tensor)


def invert_from_tensor(tensor: "Tensor") -> "Tensor":
    from .tensor_ops import _scalar_to_tensor

    return _scalar_to_tensor(1.0, tensor._dtype).sub(tensor)
