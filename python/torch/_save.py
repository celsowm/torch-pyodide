"""torch.save / torch.load — PyTorch-compatible zipfile format.

Writes and reads the same zipfile layout that real `torch.save` produces:

    archive/data.pkl         pickle of the object graph
    archive/data/<key>       raw tensor storage bytes (one file per storage)
    archive/version          "1"
    archive/.format_version  "1"
    archive/byteorder        "little"

Files saved by `torch.save` (real PyTorch) can be loaded by `torch.load`
when their content is contiguous float32 tensors (the common state_dict
case). Saving also produces a compatible file.

NOTE: contiguous float32 tensors only — no strided views, no
storage_offset, no non-float dtypes. This covers the overwhelming
majority of state_dicts.
"""

from __future__ import annotations

import io
import pickle
import struct
import zipfile
from collections import OrderedDict

from torch import Tensor


# ── Format constants (must match real torch.save) ────────────────

_SAVE_VERSION = "1"
_SAVE_FORMAT_VERSION = "1"
_SAVE_BYTEORDER = "little"


# ── Storage / rebuild shims (resolved on load) ──────────────────


class _StorageBase:
    """Base for storage shims. Carries the element size (4 = f32, 8 = i64)."""

    def __init__(self, name: str = "", elem_size: int = 4) -> None:
        self.name = name
        self._raw: bytes = b""
        self._numel: int = 0
        self._elem_size: int = elem_size

    def __setstate__(self, state: object) -> None:
        if isinstance(state, (bytes, bytearray, memoryview)):
            self._raw = bytes(state)
        elif isinstance(state, tuple) and len(state) >= 1:
            self._raw = bytes(state[0])
        else:
            self._raw = bytes(state)  # type: ignore[arg-type]
        self._numel = len(self._raw) // self._elem_size

    def __repr__(self) -> str:
        return f"_StorageBase(name={self.name!r}, numel={self._numel}, elem_size={self._elem_size})"


class _FloatStorage(_StorageBase):
    """Stand-in for `torch.FloatStorage` (little-endian f32 bytes)."""

    def __init__(self, name: str = "") -> None:
        super().__init__(name, elem_size=4)


class _LongStorage(_StorageBase):
    """Stand-in for `torch.LongStorage` (little-endian i64 bytes)."""

    def __init__(self, name: str = "") -> None:
        super().__init__(name, elem_size=8)


def _rebuild_tensor_v2(
    storage: "_StorageBase",
    storage_offset: int,
    size: tuple,
    stride: tuple,
    requires_grad: bool,
    backward_hooks: object,
) -> Tensor:
    from torch.tensor_factories_ops import tensor_from_data

    raw = storage._raw
    elem_size = getattr(storage, "_elem_size", 4)
    if storage_offset:
        raw = raw[storage_offset * elem_size :]
    n = len(raw) // elem_size
    if elem_size == 4:
        values: list = list(struct.unpack(f"<{n}f", raw[: n * 4]))
        dtype = "float32"
    elif elem_size == 8:
        values = list(struct.unpack(f"<{n}q", raw[: n * 8]))
        dtype = "int64"
    else:
        raise NotImplementedError(f"Unsupported storage element size: {elem_size}")
    size_list = list(size) if hasattr(size, "__iter__") else [int(size)]
    if not size_list:
        return tensor_from_data(values[:1], [1], dtype).reshape([])
    return tensor_from_data(values, size_list, dtype)


# ── Helpers ──────────────────────────────────────────────────────


def _tensor_to_raw_bytes(t: Tensor) -> bytes:
    """Return the tensor's contents as packed little-endian bytes.

    Supports float32 and int64 (the two dtypes produced by nn.Module
    state_dicts on BatchNorm: weight/bias/running_mean/running_var in
    float32, num_batches_tracked in int64).
    """
    dtype_str = str(t.dtype).replace("torch.", "")
    flat: list[float] | list[int] = []
    _flatten(t.tolist(), flat)
    if dtype_str == "float32":
        return struct.pack(f"<{len(flat)}f", *flat)
    if dtype_str in ("int64", "long"):
        return struct.pack(f"<{len(flat)}q", *flat)
    raise NotImplementedError(
        f"save: dtype {dtype_str!r} is not supported (only float32, int64)."
    )


def _flatten(obj, out: list) -> None:
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten(v, out)
    else:
        # Preserve int vs float so struct.pack can use the right format.
        if isinstance(obj, bool):
            out.append(int(obj))
        elif isinstance(obj, int):
            out.append(obj)
        else:
            out.append(float(obj))


def _contiguous_strides(shape: list[int]) -> tuple[int, ...]:
    """Return the contiguous (row-major) strides for a shape."""
    strides: list[int] = []
    s = 1
    for d in reversed(shape):
        strides.append(s)
        s *= int(d)
    return tuple(reversed(strides))


# ── Save ─────────────────────────────────────────────────────────


def save(obj: object, f: str | io.IOBase) -> None:
    """Serialize obj in the PyTorch zipfile format.

    The pickle stores a graph of `_FloatStorage` references. Each unique
    tensor's raw bytes are written to `archive/data/<key>`.
    """
    if isinstance(f, str):
        with open(f, "wb") as fh:
            _save_to_zip(obj, fh)
    else:
        _save_to_zip(obj, f)


class _TensorMarker:
    """Internal: replaced by the pickler with a _rebuild_tensor_v2 call."""

    __slots__ = ("key", "numel", "shape", "stride", "storage_type_name")

    def __init__(
        self,
        key: str,
        numel: int,
        shape: list[int],
        stride: tuple,
        storage_type_name: str = "FloatStorage",
    ) -> None:
        self.key = key
        self.numel = numel
        self.shape = shape
        self.stride = stride
        self.storage_type_name = storage_type_name


def _save_to_zip(obj: object, fh: io.IOBase) -> None:
    storages: list[tuple[str, bytes]] = []
    seen: dict[int, str] = {}

    def storage_key(t: Tensor) -> str:
        import sys
        # Use the runtime tensor id, not Python id(): Python's id() can be
        # reused across garbage-collected Tensor objects, which causes two
        # distinct Tensors to collide on the same storage key.
        tid = t._id
        if tid in seen:
            return seen[tid]
        key = str(len(storages))
        seen[tid] = key
        raw = _tensor_to_raw_bytes(t)
        storages.append((key, raw))
        return key

    def _maybe_state_dict_entry_to_tensor(o: object) -> object:
        """If `o` is a dict with state_dict shape {shape, data, dtype},
        materialise it as a Tensor. Otherwise return as-is.

        torch-pyodide's `Module.state_dict()` returns dicts of
        {shape, data, dtype} dicts (not Tensors) so the representation
        is JSON-serialisable. To save+load them through the standard
        PyTorch zipfile format we first convert each entry to a
        real Tensor; the rest of `wrap()` then handles it like any
        other Tensor and emits a `_rebuild_tensor_v2` call.
        """
        if not isinstance(o, dict):
            return o
        keys = set(o.keys())
        if not ({"shape", "data", "dtype"} <= keys):
            return o
        shape = list(o["shape"])
        raw_data = o["data"]
        # 0-d tensors serialise to a bare scalar (not a list) in JSON.
        # Wrap to a single-element list so the tensor constructor accepts it.
        if isinstance(raw_data, (int, float)):
            data = [raw_data]
        else:
            data = list(raw_data)
        dtype_str = str(o["dtype"]).replace("torch.", "")
        if dtype_str == "float32":
            dtype = "float32"
        elif dtype_str == "int64" or dtype_str == "long":
            dtype = "int64"
        else:
            return o
        from torch.tensor_factories_ops import tensor_from_data

        return tensor_from_data(data, shape, dtype)

    def wrap(o: object) -> object:
        """Walk the object graph and replace Tensors with markers.

        Preserves the original container type (OrderedDict, dict, list,
        tuple) so that load() returns the same type as save() received.
        """
        # Convert torch-pyodide state_dict entries (dict-of-{shape,data,dtype})
        # to real Tensors so the standard pickle path emits _rebuild_tensor_v2.
        o = _maybe_state_dict_entry_to_tensor(o)
        if isinstance(o, Tensor):
            key = storage_key(o)
            shape = [int(s) for s in o.shape]
            numel = 1
            for s in shape:
                numel *= s
            dtype_str = str(o.dtype).replace("torch.", "")
            storage_type_name = "FloatStorage" if dtype_str == "float32" else "LongStorage"
            return _TensorMarker(
                key, numel, shape, _contiguous_strides(shape), storage_type_name
            )
        if isinstance(o, dict):
            # Preserve the dict subclass (OrderedDict, defaultdict, …)
            wrapped = {k: wrap(v) for k, v in o.items()}
            return type(o)(wrapped) if type(o) is not dict else wrapped
        if isinstance(o, list):
            return [wrap(v) for v in o]
        if isinstance(o, tuple):
            return tuple(wrap(v) for v in o)
        return o

    wrapped = wrap(obj)

    # Use the pure-Python Pickler so we can override `save()` for _TensorMarker
    # and emit BINPERSID opcodes that match real torch.save output.
    class _Pickler(pickle._Pickler):
        def persistent_id(self, value):  # type: ignore[override]
            return None

        def save(self, obj, save_persistent_id=True):  # type: ignore[override]
            if isinstance(obj, _TensorMarker):
                pickler = self
                # Emit: _rebuild_tensor_v2(
                #         BINPERSID(('storage', FloatStorage, key, 'cpu', numel)),
                #         storage_offset=0,
                #         size=shape,
                #         stride=stride,
                #         requires_grad=False,
                #         backward_hooks=OrderedDict(),
                #       )
                # Note: FloatStorage is emitted as a GLOBAL (class reference),
                # not a string, so unpickling resolves it via find_class.
                pickler.write(b"ctorch._utils\n_rebuild_tensor_v2\n")
                pickler.write(pickle.MARK)  # outer args tuple
                # Build the inner storage tuple manually on the stack:
                pickler.write(pickle.MARK)
                pickler.save("storage")
                # Emit a GLOBAL opcode that resolves to torch.{Float,Long}Storage
                # (a class) on load. We write the opcode bytes directly
                # because save_global() needs the object, not a name.
                pickler.write(f"ctorch\n{obj.storage_type_name}\n".encode("ascii"))
                pickler.save(obj.key)
                pickler.save("cpu")
                pickler.save(obj.numel)
                pickler.write(pickle.TUPLE)
                # Convert the top of stack (the inner tuple) into a persistent_id:
                pickler.write(b"Q")  # BINPERSID
                # The rest of the args for _rebuild_tensor_v2:
                pickler.save(0)  # storage_offset
                pickler.save(tuple(obj.shape))  # size
                pickler.save(tuple(obj.stride))  # stride
                pickler.save(False)  # requires_grad
                pickler.save(OrderedDict())  # backward_hooks
                pickler.write(pickle.TUPLE)  # close outer args
                pickler.write(b"R")  # REDUCE
                return
            super().save(obj, save_persistent_id)

    pkl_buf = io.BytesIO()
    _Pickler(pkl_buf, protocol=2).dump(wrapped)
    pkl_data = pkl_buf.getvalue()

    with zipfile.ZipFile(fh, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("archive/version", _SAVE_VERSION)
        zf.writestr("archive/.format_version", _SAVE_FORMAT_VERSION)
        zf.writestr("archive/byteorder", _SAVE_BYTEORDER)
        zf.writestr("archive/data.pkl", pkl_data)
        for key, raw in storages:
            zf.writestr(f"archive/data/{key}", raw)


# ── Load ─────────────────────────────────────────────────────────


def load(
    f: str | io.IOBase,
    map_location: object = None,
    weights_only: bool = False,
) -> object:
    """Deserialize a file produced by `torch.save`.

    Supports the standard PyTorch zipfile layout (archive/...) for the
    contiguous float32 case. Falls back to plain pickle for files
    written by the legacy format.
    """
    if isinstance(f, str):
        with open(f, "rb") as fh:
            return _load_from_file(fh, map_location)
    return _load_from_file(f, map_location)


def _load_from_file(fh: io.IOBase, map_location: object) -> object:
    raw = fh.read()
    return _load_from_bytes(raw, map_location)


def _load_from_bytes(raw: bytes, map_location: object) -> object:
    # Try zipfile format first
    if raw[:2] == b"PK":
        return _load_zip_bytes(raw, map_location)
    # Fallback: legacy pickle
    return _LegacyUnpickler(io.BytesIO(raw), map_location).load()


def _load_zip_bytes(raw: bytes, map_location: object) -> object:
    zf = zipfile.ZipFile(io.BytesIO(raw))
    names = zf.namelist()
    if "archive/data.pkl" not in names:
        raise RuntimeError(
            "Invalid torch.save archive: missing archive/data.pkl"
        )
    pkl_bytes = zf.read("archive/data.pkl")
    storage_data: dict[str, bytes] = {}
    for name in names:
        if name.startswith("archive/data/") and not name.endswith("/"):
            key = name[len("archive/data/") :]
            storage_data[key] = zf.read(name)
    return _ArchiveUnpickler(
        io.BytesIO(pkl_bytes), storage_data, map_location
    ).load()


class _ArchiveUnpickler(pickle.Unpickler):
    def __init__(
        self,
        file: io.IOBase,
        storage_data: dict[str, bytes],
        map_location: object,
    ) -> None:
        super().__init__(file)
        self._storage_data = storage_data
        self._map_location = map_location
        self.find_class = self._patched_find_class  # type: ignore[assignment]

    def _patched_find_class(self, module: str, name: str):
        if module == "torch" and name == "FloatStorage":
            return _FloatStorage
        if module == "torch" and name == "LongStorage":
            return _LongStorage
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        return super().find_class(module, name)

    def persistent_load(self, pid):  # type: ignore[override]
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"Unsupported persistent id: {pid!r}")
        key = pid[2]
        # Use the storage class from the pid so elem_size / dtype are right.
        storage_cls = pid[1]
        if isinstance(storage_cls, type) and issubclass(storage_cls, _StorageBase):
            storage = storage_cls(name=key)
        else:
            storage = _FloatStorage(name=key)
        raw = self._storage_data.get(key)
        if raw is None:
            raise RuntimeError(f"Missing storage data for key {key!r}")
        storage.__setstate__(raw)
        return storage


class _LegacyUnpickler(pickle.Unpickler):
    """Loads the old (pre-zipfile) format produced by older torch-pyodide."""

    def __init__(self, file: io.IOBase, map_location: object) -> None:
        super().__init__(file)
        self._map_location = map_location


# ── jit stubs (not implemented) ──────────────────────────────────


def jit_save(obj: object, f: str) -> None:
    save(obj, f)


def jit_load(f: str) -> object:
    return load(f)
