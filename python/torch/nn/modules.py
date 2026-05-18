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

    def register_parameter(self, name: str, param: Tensor | None) -> None:
        if param is not None:
            self._parameters[name] = param

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
            self.register_parameter(name, value)
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
            self.bias = None  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        from .functional import conv2d
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)


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
