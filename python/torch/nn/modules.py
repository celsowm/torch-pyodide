from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from torch._tensor import softmax_from_tensor, log_softmax_from_tensor


# ── Base Module ───────────────────────────────────────────────────

class Module:
    _parameters: dict[str, Tensor]
    _modules: dict[str, Module]
    training: bool

    def __init__(self) -> None:
        self._parameters = {}
        self._modules = {}
        self.training = True

    def register_parameter(self, name: str, param: Tensor | None) -> Tensor | None:
        if param is not None:
            if not isinstance(param, Parameter):
                param = Parameter(param)
            self._parameters[name] = param
        return param

    def register_module(self, name: str, module: Module) -> None:
        self._modules[name] = module

    def parameters(self) -> list[Tensor]:
        params = list(self._parameters.values())
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def train(self, mode: bool = True) -> Module:
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def __call__(self, *args: object, **kwargs: object) -> Tensor:
        return self.forward(*args, **kwargs)  # type: ignore

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        raise NotImplementedError

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, Tensor):
            value = self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.register_module(name, value)
        super().__setattr__(name, value)


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
        self.running_mean = torch.zeros((num_features,))
        self.running_var = torch.ones((num_features,))

    def forward(self, x: Tensor) -> Tensor:
        from .functional import batch_norm
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
        self.weight = torch.randn((out_channels, in_channels, kernel_size)) * 0.01
        if bias:
            self.bias = torch.zeros((out_channels,))
        else:
            self.bias = None

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
        self.weight = torch.randn((out_channels, in_channels, kernel_size[0], kernel_size[1])) * 0.01
        if bias:
            self.bias = torch.zeros((out_channels,))
        else:
            self.bias = None

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
        self.weight = torch.randn((in_channels, out_channels, kernel_size[0], kernel_size[1])) * 0.01
        if bias:
            self.bias = torch.zeros((out_channels,))
        else:
            self.bias = None

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
        self.weight = torch.randn((num_embeddings, embedding_dim)) * 0.01

    def forward(self, x: Tensor) -> Tensor:
        # index_select based lookup
        result = torch.index_select(self.weight, 0, x)
        result = result.reshape(list(x.shape) + [self.embedding_dim])
        if self.padding_idx is not None:
            mask = x == self.padding_idx
            if isinstance(mask, Tensor):
                result = result.masked_fill(mask.unsqueeze(-1).expand(*result.shape), 0.0)
        return result


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
