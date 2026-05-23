from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._tensor import Tensor


def split_from_tensor(tensor: "Tensor", split_size: int | list[int], dim: int = 0) -> list["Tensor"]:
    shape = tensor._shape
    d = dim if dim >= 0 else dim + len(shape)
    size_dim = shape[d]
    if isinstance(split_size, int):
        sections = []
        i = 0
        while i < size_dim:
            end = min(i + split_size, size_dim)
            sections.append(end - i)
            i = end
    else:
        sections = [int(s) for s in split_size]
    result: list[Tensor] = []
    offset = 0
    for sec in sections:
        result.append(tensor.slice(dim=d, start=offset, end=offset + sec))
        offset += sec
    return result


def chunk_from_tensor(tensor: "Tensor", chunks: int, dim: int = 0) -> list["Tensor"]:
    shape = tensor._shape
    d = dim if dim >= 0 else dim + len(shape)
    size_dim = shape[d]
    split_size = (size_dim + chunks - 1) // chunks
    return split_from_tensor(tensor, split_size, dim=dim)
