from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from torch._tensor import softmax_from_tensor, log_softmax_from_tensor


# ── Base Module ───────────────────────────────────────────────────

class Module:
    _parameters: dict[str, Tensor]
    _modules: dict[str, Module]
    _buffers: dict[str, Tensor]
    training: bool

    def __init__(self) -> None:
        self._parameters = {}
        self._modules = {}
        self._buffers = {}
        self.training = True

    def register_parameter(self, name: str, param: Tensor | None) -> Tensor | None:
        if param is not None:
            if not isinstance(param, Parameter):
                param = Parameter(param)
            self._parameters[name] = param
        return param

    def register_module(self, name: str, module: Module) -> None:
        self._modules[name] = module

    def register_buffer(self, name: str, tensor: Tensor | None, persistent: bool = True) -> Tensor | None:
        """Register a buffer (non-parameter persistent state)."""
        if tensor is not None:
            self._buffers[name] = tensor
        return tensor

    def parameters(self) -> list[Tensor]:
        params = list(self._parameters.values())
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> list[tuple[str, Tensor]]:
        """Yield (qualified_name, parameter) pairs, recursing into submodules."""
        out: list[tuple[str, Tensor]] = []
        for name, p in self._parameters.items():
            full = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            out.append((full, p))
        if recurse:
            for mod_name, m in self._modules.items():
                sub_prefix = f"{prefix}{mod_name}" if not prefix else f"{prefix}.{mod_name}"
                out.extend(m.named_parameters(prefix=sub_prefix, recurse=True))
        return out

    def named_modules(self, prefix: str = "", recurse: bool = True) -> list[tuple[str, "Module"]]:
        """Yield (qualified_name, module) pairs, including self with empty prefix."""
        out: list[tuple[str, Module]] = []
        out.append((prefix, self))
        if recurse:
            for name, m in self._modules.items():
                sub_prefix = name if not prefix else f"{prefix}.{name}"
                out.extend(m.named_modules(prefix=sub_prefix, recurse=True))
        return out

    def buffers(self, recurse: bool = True) -> list[Tensor]:
        out = list(self._buffers.values())
        if recurse:
            for m in self._modules.values():
                out.extend(m.buffers(recurse=True))
        return out

    def named_buffers(self, prefix: str = "", recurse: bool = True) -> list[tuple[str, Tensor]]:
        out: list[tuple[str, Tensor]] = []
        for name, b in self._buffers.items():
            full = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            out.append((full, b))
        if recurse:
            for mod_name, m in self._modules.items():
                sub_prefix = f"{prefix}{mod_name}" if not prefix else f"{prefix}.{mod_name}"
                out.extend(m.named_buffers(prefix=sub_prefix, recurse=True))
        return out

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False) -> dict[str, object]:
        """Return a serializable state dictionary of parameters and buffers.

        Each entry is {shape: list[int], data: list[float], dtype: str}.
        Compatible with load_state_dict (round-trip).
        """
        out: dict[str, object] = {} if destination is None else destination
        for name, p in self._parameters.items():
            full = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            if keep_vars:
                out[full] = p
            else:
                out[full] = _serialize_tensor(p)
        for name, b in self._buffers.items():
            full = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            if keep_vars:
                out[full] = b
            else:
                out[full] = _serialize_tensor(b)
        for mod_name, m in self._modules.items():
            sub_prefix = mod_name if not prefix else f"{prefix}.{mod_name}"
            m.state_dict(destination=out, prefix=sub_prefix, keep_vars=keep_vars)
        return out

    def load_state_dict(self, state_dict: dict[str, object], strict: bool = True) -> object:
        """Load a state_dict previously produced by self.state_dict().

        Returns an _IncompatibleKeys-like object with .missing_keys and
        .unexpected_keys (always empty for strict=False; for strict=True
        raises on mismatch).
        """
        missing: list[str] = []
        unexpected: list[str] = []
        # Build a flat name -> Module map for quick lookup
        named_params = dict(self.named_parameters())
        named_buffers = dict(self.named_buffers())
        all_known = set(named_params) | set(named_buffers)
        seen: set[str] = set()
        for key, value in state_dict.items():
            if key in named_params:
                _load_into_tensor(named_params[key], value)
                seen.add(key)
            elif key in named_buffers:
                _load_into_tensor(named_buffers[key], value)
                seen.add(key)
            else:
                unexpected.append(key)
        for key in all_known:
            if key not in seen:
                missing.append(key)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Error(s) in loading state_dict for {type(self).__name__}: "
                f"missing keys: {missing}; unexpected keys: {unexpected}"
            )
        return _IncompatibleKeys(missing, unexpected)

    def apply(self, fn: "Callable[[Module], Module]") -> "Module":
        """Apply fn to each submodule (including self) — children-first, then self.
        Matches PyTorch's depth-first pre-order traversal semantics.
        """
        for _, child in self._modules.items():
            child.apply(fn)
        fn(self)
        return self

    def to(self, device=None, dtype=None, non_blocking: bool = False) -> "Module":
        """Move/cast all parameters and buffers. Only CPU and float32 are supported."""
        # We are CPU-only; only honor dtype for the float32 case.
        if dtype is not None and str(dtype) != "float32":
            raise NotImplementedError(
                f"dtype {dtype!r} is not supported by this runtime "
                f"(only float32 / torch.float32 is available)."
            )
        if device is not None and str(device) != "cpu":
            raise RuntimeError(f"Only CPU device is supported, got: {device!r}")
        return self

    def cpu(self) -> "Module":
        return self.to("cpu")

    def cuda(self, device: int | None = None) -> "Module":
        # Stub: no GPU device API for now; return self.
        return self

    def double(self) -> "Module":
        raise NotImplementedError(
            "float64 is not supported by this runtime; only float32 is available."
        )

    def float(self) -> "Module":
        return self.to(dtype="float32")

    def train(self, mode: bool = True) -> Module:
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def requires_grad_(self, requires_grad: bool = True) -> Module:
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.parameters():
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                import torch
                p.grad = torch.zeros_like(p.grad)

    def __call__(self, *args: object, **kwargs: object) -> Tensor:
        return self.forward(*args, **kwargs)  # type: ignore

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        raise NotImplementedError

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, Tensor):
            # If this tensor is already a registered buffer, don't re-register
            # it as a parameter; just rebind the attribute to the same object.
            if name in self._buffers:
                self._buffers[name] = value
            else:
                value = self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.register_module(name, value)
        super().__setattr__(name, value)


# ── Serialization helpers ──────────────────────────────────────────

class _IncompatibleKeys:
    def __init__(self, missing: list[str], unexpected: list[str]) -> None:
        self.missing_keys = missing
        self.unexpected_keys = unexpected

    def __repr__(self) -> str:
        return f"_IncompatibleKeys(missing_keys={self.missing_keys}, unexpected_keys={self.unexpected_keys})"


def _serialize_tensor(t: Tensor) -> dict[str, object]:
    """Snapshot a tensor into a JSON-serializable dict."""
    return {
        "shape": list(t.shape),
        "data": t.tolist(),
        "dtype": t.dtype,
    }


def _load_into_tensor(target: Tensor, value: object) -> None:
    """Replace a tensor's storage with serialized data (shape + flat values).

    Accepts both the torch-pyodide native state_dict format
    ({shape, data, dtype} dicts) and real-PyTorch format
    (raw Tensors, e.g. returned by `torch.load`).
    """
    from torch import Tensor as _Tensor

    if isinstance(value, _Tensor):
        # Real-PyTorch format: a live Tensor. Copy its data into target.
        flat = value.tolist()
        if not isinstance(flat, list):
            flat = [flat]
        target_shape = list(value.shape)
        target_dtype = str(value.dtype).replace("torch.", "")
        import torch
        new_t = torch.tensor(flat, dtype=target_dtype)
        if list(new_t.shape) != target_shape:
            new_t = new_t.reshape(target_shape)
        target._set(new_t)
        return
    if not isinstance(value, dict) or "shape" not in value or "data" not in value:
        raise RuntimeError(f"Cannot load state value into tensor: {value!r}")
    target_shape = list(value["shape"])
    target_data = value["data"]
    # 0-d tensors (scalars) come out of `tolist()` as a bare scalar
    # rather than a 1-element list. Wrap so `torch.tensor(...)` produces
    # a 0-d tensor that we can then `.reshape` to the target shape.
    if not isinstance(target_data, list):
        target_data = [target_data]
    target_dtype = value.get("dtype", target.dtype)
    import torch
    new_t = torch.tensor(target_data, dtype=target_dtype)
    # Reshape to the recorded shape
    if list(new_t.shape) != target_shape:
        new_t = new_t.reshape(target_shape)
    target._set(new_t)


# ── Containers ────────────────────────────────────────────────────

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        for i, m in enumerate(modules):
            self.register_module(str(i), m)

    def forward(self, x: Tensor) -> Tensor:
        for m in self._modules.values():
            x = m(x)
        return x

    def __getitem__(self, idx: int) -> Module:
        return self._modules[str(idx)]

    def __len__(self) -> int:
        return len(self._modules)


class ModuleList(Module):
    """Holds submodules in a list. Behaves like a Python list."""
    def __init__(self, modules=None) -> None:
        super().__init__()
        self._list: list[Module] = []
        if modules is not None:
            for m in modules:
                self.append(m)

    def append(self, module: Module) -> "ModuleList":
        idx = len(self._list)
        self._list.append(module)
        self.register_module(str(idx), module)
        return self

    def extend(self, modules) -> "ModuleList":
        for m in modules:
            self.append(m)
        return self

    def insert(self, index: int, module: Module) -> None:
        self._list.insert(index, module)
        self._rebuild_dict()

    def __getitem__(self, idx: int) -> Module:
        return self._list[idx]

    def __setitem__(self, idx: int, module: Module) -> None:
        self._list[idx] = module
        self._rebuild_dict()

    def __delitem__(self, idx: int) -> None:
        del self._list[idx]
        self._rebuild_dict()

    def __len__(self) -> int:
        return len(self._list)

    def __iter__(self):
        return iter(self._list)

    def __repr__(self) -> str:
        lines = [f"  ({i}): {m}" for i, m in enumerate(self._list)]
        return "ModuleList(\n" + "\n".join(lines) + "\n)"

    def _rebuild_dict(self) -> None:
        self._modules.clear()
        for i, m in enumerate(self._list):
            self._modules[str(i)] = m


class ModuleDict(Module):
    """Holds submodules in a dictionary."""
    def __init__(self, modules=None) -> None:
        super().__init__()
        if modules is not None:
            for name, m in modules.items():
                self[name] = m

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.register_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __repr__(self) -> str:
        lines = [f"  ({k}): {m}" for k, m in self._modules.items()]
        return "ModuleDict(\n" + "\n".join(lines) + "\n)"


# ── Parameter ─────────────────────────────────────────────────────

class Parameter(Tensor):
    """A Tensor that is automatically registered as a parameter when assigned to a Module."""
    def __new__(cls, data=None, requires_grad=True, dtype="float32"):
        import torch
        if data is None:
            data = torch.tensor([], dtype=dtype)
        if isinstance(data, list):
            import torch
            data = torch.tensor(data, dtype=dtype)
        instance = object.__new__(cls)
        instance._id = data._id
        instance._shape = list(data.shape)
        instance._dtype = data.dtype
        instance._requires_grad = requires_grad
        instance._node = None
        instance._backward_hooks = {}
        instance.grad = None
        instance._retains_grad = False
        instance._data = data
        return instance

    def __init__(self, data=None, requires_grad=True, dtype="float32"):
        pass

    def __repr__(self) -> str:
        return f"Parameter containing:\n{super().__repr__()}"


class ParameterList(Module):
    """Holds parameters in a list."""
    def __init__(self, parameters=None) -> None:
        super().__init__()
        self._params: list[Parameter] = []
        if parameters is not None:
            for p in parameters:
                self.append(p)

    def append(self, param: Parameter) -> "ParameterList":
        idx = len(self._params)
        self._params.append(param)
        self.register_parameter(str(idx), param)
        return self

    def extend(self, parameters) -> "ParameterList":
        for p in parameters:
            self.append(p)
        return self

    def __getitem__(self, idx: int) -> Parameter:
        return self._params[idx]

    def __setitem__(self, idx: int, param: Parameter) -> None:
        self._params[idx] = param

    def __iter__(self):
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)


class ParameterDict(Module):
    """Holds parameters in a dictionary."""
    def __init__(self, parameters=None) -> None:
        super().__init__()
        if parameters is not None:
            for name, p in parameters.items():
                self[name] = p

    def __getitem__(self, key: str) -> Parameter:
        return self._parameters[key]

    def __setitem__(self, key: str, param: Parameter) -> None:
        self.register_parameter(key, param)

    def keys(self):
        return self._parameters.keys()

    def values(self):
        return self._parameters.values()

    def items(self):
        return self._parameters.items()

    def __len__(self) -> int:
        return len(self._parameters)


# ── Linear ────────────────────────────────────────────────────────

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty((out_features, in_features))
        if bias:
            self.bias = torch.empty((out_features,))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import math
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 1
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        from .functional import linear
        return linear(x, self.weight, self.bias)


# ── Bilinear ──────────────────────────────────────────────────────

class Bilinear(Module):
    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = torch.randn((out_features, in1_features, in2_features)) * 0.01
        if bias:
            self.bias = torch.zeros((out_features,))
        else:
            self.bias = None

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        from .functional import bilinear
        return bilinear(x1, x2, self.weight, self.bias)


# ── Dropout ───────────────────────────────────────────────────────

class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        from .functional import dropout
        return dropout(x, self.p, self.training)


class Dropout2d(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        from .functional import dropout2d
        return dropout2d(x, self.p, self.training)


# ── Normalization ─────────────────────────────────────────────────

class _BatchNorm(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = torch.ones((num_features,))
        self.bias = torch.zeros((num_features,))
        # running stats are buffers (not parameters)
        self.register_buffer("running_mean", torch.zeros((num_features,)))
        self.register_buffer("running_var", torch.ones((num_features,)))
        # `num_batches_tracked` is a scalar int64 buffer that real PyTorch
        # carries in the state_dict. It increments on every forward pass
        # in training mode and is not used in eval mode. Initialise to 0.
        self.register_buffer("num_batches_tracked", torch.zeros((), dtype="int64"))
        # Re-attach as attributes for downstream functional calls (the
        # __setattr__ short-circuited to register_parameter when we used
        # `self.running_mean = ...` directly).
        self.running_mean = self._buffers["running_mean"]
        self.running_var = self._buffers["running_var"]
        self.num_batches_tracked = self._buffers["num_batches_tracked"]

    def forward(self, x: Tensor) -> Tensor:
        from .functional import batch_norm
        if self.training:
            # Increment num_batches_tracked (int64 0-d buffer) in-place.
            # Real PyTorch does this every forward pass in training mode.
            # `tolist()` on a 0-d tensor returns a bare Python int, not a list.
            cur = self.num_batches_tracked.tolist()
            if isinstance(cur, list):
                cur = cur[0] if cur else 0
            self.num_batches_tracked = self.num_batches_tracked + 1
        return batch_norm(
            x,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )


class BatchNorm1d(_BatchNorm):
    pass


class BatchNorm2d(_BatchNorm):
    pass


class LayerNorm(Module):
    def __init__(self, normalized_shape: int | Sequence[int], eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = torch.ones(list(normalized_shape))
        self.bias = torch.zeros(list(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        from .functional import layer_norm
        return layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class GroupNorm(Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(
                f"GroupNorm: num_channels ({num_channels}) must be divisible by "
                f"num_groups ({num_groups})"
            )
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = torch.ones((num_channels,))
            self.bias = torch.zeros((num_channels,))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        from .functional import group_norm
        return group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class InstanceNorm1d(GroupNorm):
    """Instance normalization for 2D/3D inputs (N, C, L) — groups=channels."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False,
                 track_running_stats: bool = False):
        # InstanceNorm with affine=False is the most common default; with affine=True
        # we still expose weight/bias of shape (C,).
        super().__init__(num_groups=num_features, num_channels=num_features, eps=eps, affine=affine)
        self.num_features = num_features
        self.track_running_stats = track_running_stats

    def forward(self, x: Tensor) -> Tensor:
        from .functional import group_norm
        return group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class InstanceNorm2d(GroupNorm):
    """Instance normalization for 4D inputs (N, C, H, W) — groups=channels."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False,
                 track_running_stats: bool = False):
        super().__init__(num_groups=num_features, num_channels=num_features, eps=eps, affine=affine)
        self.num_features = num_features
        self.track_running_stats = track_running_stats

    def forward(self, x: Tensor) -> Tensor:
        from .functional import group_norm
        return group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


# ── Activations ───────────────────────────────────────────────────

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.silu()


class LeakyReLU(Module):
    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return x.leaky_relu(self.alpha)


class PReLU(Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = torch.tensor([init] * num_parameters)

    def forward(self, x: Tensor) -> Tensor:
        from .functional import prelu
        return prelu(x, self.weight)


class ELU(Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        from .functional import elu
        return elu(x, self.alpha)


class CELU(Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        from .functional import celu
        return celu(x, self.alpha)


class RReLU(Module):
    def __init__(self, lower: float = 0.125, upper: float = 0.3333333333333333) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper

    def forward(self, x: Tensor) -> Tensor:
        from .functional import rrelu
        return rrelu(x, self.lower, self.upper, self.training)


class GLU(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        from .functional import glu
        return glu(x, self.dim)


class Softmax(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return softmax_from_tensor(x, self.dim)


class LogSoftmax(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return log_softmax_from_tensor(x, self.dim)


# ── Shape ─────────────────────────────────────────────────────────

class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(start_dim=self.start_dim, end_dim=self.end_dim)


class Unflatten(Module):
    def __init__(self, dim: int, unflattened_size: Sequence[int]) -> None:
        super().__init__()
        self.dim = dim
        self.unflattened_size = list(unflattened_size)

    def forward(self, x: Tensor) -> Tensor:
        shape = list(x.shape)
        dim = self.dim if self.dim >= 0 else self.dim + len(shape)
        new_shape = shape[:dim] + self.unflattened_size + shape[dim + 1:]
        return x.reshape(new_shape)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


# ── Convolution ───────────────────────────────────────────────────

class Conv1d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = torch.empty((out_channels, in_channels, kernel_size))
        if bias:
            self.bias = torch.empty((out_channels,))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import math
        import torch.nn.init as _init
        _init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            _init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # Convert 1D conv to 2D conv: add a dummy H dimension (H=1)
        # Input: (N, C, L) -> (N, C, 1, L)
        x_2d = x.unsqueeze(2)
        w_2d = self.weight.unsqueeze(2)
        from .functional import conv2d
        result_2d = conv2d(x_2d, w_2d, self.bias, self.stride, self.padding)
        return result_2d.squeeze(2)


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = torch.empty((out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = torch.empty((out_channels,))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import math
        import torch.nn.init as _init
        _init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            _init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        from .functional import conv2d
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)


class ConvTranspose2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = torch.empty((in_channels, out_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = torch.empty((out_channels,))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import math
        import torch.nn.init as _init
        _init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            _init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        from .functional import conv2d
        return conv2d(x, self.weight.transpose(0, 1), self.bias, 1, 0)


# ── Embedding ─────────────────────────────────────────────────────

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = torch.empty((num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as _init
        _init.normal_(self.weight, mean=0.0, std=1.0)
        if self.padding_idx is not None and self.padding_idx < self.num_embeddings:
            with torch.no_grad():
                row_ids = torch.arange(self.num_embeddings, dtype=torch.long).unsqueeze(1).expand_as(self.weight)
                pad_mask = (row_ids == self.padding_idx)
                self.weight._set(self.weight.masked_fill(pad_mask, 0.0))

    def forward(self, x: Tensor) -> Tensor:
        from torch.tensor_nn_ops import embedding_from_tensor
        if self.padding_idx is not None:
            with torch.no_grad():
                row_ids = torch.arange(self.num_embeddings, dtype=torch.long).unsqueeze(1).expand_as(self.weight)
                pad_mask = (row_ids == self.padding_idx)
                self.weight._set(self.weight.masked_fill(pad_mask, 0.0))
        return embedding_from_tensor(self.weight, x, self.padding_idx if self.padding_idx is not None else -1)


# ── Pooling ───────────────────────────────────────────────────────

class MaxPool2d(Module):
    def __init__(self, kernel_size: int | tuple[int, int], stride: int | None = None, padding: int = 0) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        from .functional import max_pool2d
        return max_pool2d(x, self.kernel_size, self.stride, self.padding)


class AvgPool2d(Module):
    def __init__(self, kernel_size: int | tuple[int, int], stride: int | None = None, padding: int = 0) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        from .functional import avg_pool2d
        return avg_pool2d(x, self.kernel_size, self.stride, self.padding)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size: int | tuple[int, int]) -> None:
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2], x.shape[3]
        oh, ow = self.output_size
        stride_h = h // oh
        stride_w = w // ow
        kernel_h = h - (oh - 1) * stride_h
        kernel_w = w - (ow - 1) * stride_w
        from .functional import avg_pool2d
        return avg_pool2d(x, (kernel_h, kernel_w), (stride_h, stride_w), 0)


class AdaptiveMaxPool2d(Module):
    def __init__(self, output_size: int | tuple[int, int]) -> None:
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2], x.shape[3]
        oh, ow = self.output_size
        stride_h = h // oh
        stride_w = w // ow
        kernel_h = h - (oh - 1) * stride_h
        kernel_w = w - (ow - 1) * stride_w
        from .functional import max_pool2d
        return max_pool2d(x, (kernel_h, kernel_w), (stride_h, stride_w), 0)


# ── Pooling 1D ───────────────────────────────────────────────────

class MaxPool1d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        from .functional import max_pool1d
        return max_pool1d(x, self.kernel_size, self.stride, self.padding)


class AvgPool1d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        from .functional import avg_pool1d
        return avg_pool1d(x, self.kernel_size, self.stride, self.padding)


class ReplicationPad1d(Module):
    def __init__(self, padding: int | tuple[int, int]) -> None:
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        from .functional import pad
        return pad(x, self.padding, mode="replicate")


class ReplicationPad2d(Module):
    def __init__(self, padding: int | tuple[int, int] | tuple[int, int, int, int]) -> None:
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        self.padding = tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        from .functional import pad
        return pad(x, self.padding, mode="replicate")


class ReflectionPad1d(Module):
    def __init__(self, padding: int | tuple[int, int]) -> None:
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        from .functional import pad
        return pad(x, self.padding, mode="reflect")


class ReflectionPad2d(Module):
    def __init__(self, padding: int | tuple[int, int] | tuple[int, int, int, int]) -> None:
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        self.padding = tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        from .functional import pad
        return pad(x, self.padding, mode="reflect")


class CircularPad1d(Module):
    def __init__(self, padding: int | tuple[int, int]) -> None:
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        from .functional import pad
        return pad(x, self.padding, mode="circular")


class CircularPad2d(Module):
    def __init__(self, padding: int | tuple[int, int] | tuple[int, int, int, int]) -> None:
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        self.padding = tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        from .functional import pad
        return pad(x, self.padding, mode="circular")


class ConstantPad1d(Module):
    def __init__(self, padding: int | tuple[int, int], value: float = 0.0) -> None:
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = tuple(padding)
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        from .functional import pad
        return pad(x, self.padding, mode="constant", value=self.value)


class ConstantPad2d(Module):
    def __init__(self, padding: int | tuple[int, int] | tuple[int, int, int, int], value: float = 0.0) -> None:
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        self.padding = tuple(padding)
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        from .functional import pad
        return pad(x, self.padding, mode="constant", value=self.value)


class ZeroPad2d(Module):
    def __init__(self, padding: int | tuple[int, int] | tuple[int, int, int, int]) -> None:
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        self.padding = tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        from .functional import pad
        return pad(x, self.padding, mode="constant", value=0.0)


class AdaptiveAvgPool1d(Module):
    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[2]
        out_len = self.output_size
        stride = length // out_len
        kernel = length - (out_len - 1) * stride
        from .functional import avg_pool1d
        return avg_pool1d(x, kernel, stride, 0)


class Upsample(Module):
    """Upsampling layer."""
    def __init__(
        self,
        size: int | tuple[int, int] | None = None,
        scale_factor: float | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: Tensor) -> Tensor:
        from .functional import interpolate
        return interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)


# ── Loss modules ─────────────────────────────────────────────────

class CrossEntropyLoss(Module):
    """Cross entropy loss module."""
    def __init__(self, reduction: str = "mean", ignore_index: int = -100) -> None:
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        from .functional import cross_entropy
        return cross_entropy(input, target, self.reduction)


class BCEWithLogitsLoss(Module):
    """Binary cross entropy with logits module."""
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        from .functional import binary_cross_entropy_with_logits
        return binary_cross_entropy_with_logits(input, target, reduction=self.reduction)


class MSELoss(Module):
    """Mean squared error loss module."""
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        from .functional import mse_loss
        return mse_loss(input, target, reduction=self.reduction)


class L1Loss(Module):
    """L1 loss module."""
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        from .functional import l1_loss
        return l1_loss(input, target, reduction=self.reduction)


class SmoothL1Loss(Module):
    """Smooth L1 loss module (Huber with `beta` transition point)."""
    def __init__(self, reduction: str = "mean", beta: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.beta = float(beta)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        from .functional import smooth_l1_loss
        return smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)


class HuberLoss(Module):
    """Huber loss module: smooth L1 with `delta` transition (default delta=1.0).

    Equivalent to `SmoothL1Loss(beta=delta, reduction='mean')` with the
    matching defaults. We expose the same name as real PyTorch.
    """
    def __init__(self, reduction: str = "mean", delta: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.delta = float(delta)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        from .functional import smooth_l1_loss
        return smooth_l1_loss(input, target, reduction=self.reduction, beta=self.delta)
