from __future__ import annotations

import pickle
import io

from torch import Tensor


def save(obj: object, f: str | io.IOBase) -> None:
    data = _serialize(obj)
    if isinstance(f, str):
        with open(f, "wb") as fh:
            pickle.dump(data, fh)
    else:
        pickle.dump(data, f)


def load(f: str | io.IOBase, map_location: object = None, weights_only: bool = False) -> object:
    if isinstance(f, str):
        with open(f, "rb") as fh:
            data = pickle.load(fh)
    else:
        data = pickle.load(f)
    return _deserialize(data)


def _serialize(obj: object) -> object:
    if isinstance(obj, Tensor):
        return {"__tensor__": True, "data": obj.tolist(), "shape": list(obj.shape), "dtype": obj.dtype}
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


def _deserialize(obj: object) -> object:
    if isinstance(obj, dict):
        if obj.get("__tensor__"):
            from torch.tensor_factories_ops import tensor_from_data
            return tensor_from_data(obj["data"], obj["shape"], obj["dtype"])
        return {k: _deserialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserialize(v) for v in obj]
    return obj


def jit_save(obj: object, f: str) -> None:
    save(obj, f)


def jit_load(f: str) -> object:
    return load(f)
