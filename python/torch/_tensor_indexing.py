from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._tensor import Tensor


def getitem_from_tensor(tensor: "Tensor", key: object) -> object:
    if isinstance(key, int):
        return tensor.select(0, key)
    if isinstance(key, slice):
        return tensor.slice(0, key.start, key.stop, 1 if key.step is None else int(key.step))
    if isinstance(key, tensor.__class__) and key._dtype == "bool":
        from .tensor_ops import masked_select_from_tensor
        return masked_select_from_tensor(tensor, key)
    if isinstance(key, tuple):
        result = tensor
        for i, k in enumerate(key):
            if isinstance(k, int):
                result = result.select(dim=i, index=k)
            elif isinstance(k, slice):
                result = result.slice(dim=i, start=k.start, end=k.stop, step=k.step or 1)
            elif isinstance(k, tensor.__class__):
                if result.ndim <= 2:
                    result = result.index_select(dim=i, index=k.flatten())
                else:
                    from .__init__ import cat
                    indices_flat = k.flatten()
                    picked: list[Tensor] = []
                    for j in range(indices_flat._shape[0]):
                        picked.append(result.select(dim=i, index=int(indices_flat.select(0, j).item())))
                    result = cat(picked, dim=i)
            else:
                raise TypeError(f"Unsupported index type: {type(k)}")
        return result
    raise TypeError("Tensor indexing supports only int, slice, tuple, or bool Tensor in MVP.")
