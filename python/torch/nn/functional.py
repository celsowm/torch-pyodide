from __future__ import annotations

from typing import Sequence

import torch
from torch._tensor import Tensor, tensor_from_data


# ── Activation functions ──────────────────────────────────────────

def relu(x: Tensor) -> Tensor:
    return x.relu()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def gelu(x: Tensor) -> Tensor:
    return x.gelu()


def silu(x: Tensor) -> Tensor:
    return x.silu()


def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    return x.leaky_relu(alpha)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return x.softmax(dim)


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    return x.log_softmax(dim)


# ── Dropout ───────────────────────────────────────────────────────

def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training or p == 0.0:
        return x
    shape = list(x.shape)
    raw = x.tolist()
    import random as _random
    _rng = _random.Random(42)
    flat_vals = _flatten(raw)
    out_flat = []
    for v in flat_vals:
        if _rng.random() < p:
            out_flat.append(0.0)
        else:
            out_flat.append(float(v) / (1.0 - p))
    return _unflatten(shape, out_flat, x.dtype)


def dropout2d(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training or p == 0.0:
        return x
    shape = list(x.shape)
    raw = x.tolist()
    import random as _random
    _rng = _random.Random(42)
    batch, channels = shape[0], shape[1]
    flat_vals = _flatten(raw)
    out_flat = list(flat_vals)
    elem_per_channel = len(flat_vals) // (batch * channels)
    import math
    for b in range(batch):
        for c in range(channels):
            if _rng.random() < p:
                for i in range(elem_per_channel):
                    idx = b * channels * elem_per_channel + c * elem_per_channel + i
                    out_flat[idx] = 0.0
            else:
                for i in range(elem_per_channel):
                    idx = b * channels * elem_per_channel + c * elem_per_channel + i
                    out_flat[idx] = float(out_flat[idx]) / (1.0 - p)
    return _unflatten(shape, out_flat, x.dtype)


# ── Linear ────────────────────────────────────────────────────────

def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    result = x.matmul(weight.T)
    if bias is not None:
        result = result + bias
    return result


def bilinear(
    x1: Tensor, x2: Tensor, weight: Tensor, bias: Tensor | None = None
) -> Tensor:
    """Bilinear transformation using fused matmul.

    weight: (out_features, in1_features, in2_features)
    x1: (*, in1_features)
    x2: (*, in2_features)

    Uses: output[o] = sum_i x1[i] * (weight[o,i,:] @ x2)
    which avoids materializing the full (..., O, I1, I2) tensor.

    Implementation uses reshape and matmul for GPU speed:
    weight -> (out_features * in1_features, in2_features)
    x2 @ weight.T -> (batch, out_features * in1_features)
    then reshape to (batch, out_features, in1_features)
    then elementwise mul with x1 and sum over in1_features.
    """
    out_features, in1_features, in2_features = weight.shape
    orig_shape = list(x1.shape[:-1]) + [out_features]
    x1_flat = x1.reshape(-1, in1_features)
    x2_flat = x2.reshape(-1, in2_features)
    batch = x1_flat.shape[0]

    # Flatten weight: (out_features, in1_features, in2_features) -> (out_features * in1_features, in2_features)
    w_flat = weight.reshape(out_features * in1_features, in2_features)
    # x2 @ w_flat.T: (batch, out_features * in1_features)
    wx2 = x2_flat.matmul(w_flat.T)
    # Reshape to (batch, out_features, in1_features)
    wx2 = wx2.reshape(batch, out_features, in1_features)
    # x1_flat[:, None, :] * wx2: broadcast over out_features
    result = (x1_flat[:, None, :] * wx2).sum(dim=2)

    if bias is not None:
        result = result + bias

    return result.reshape(*orig_shape)


# ── Normalization ─────────────────────────────────────────────────

def batch_norm(
    x: Tensor,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    from torch._tensor import batch_norm_inference_from_tensor
    if training:
        mean = x.mean(dim=0)
        var = ((x - mean) ** 2).mean(dim=0)
        if running_mean is not None:
            running_mean._set(running_mean * (1 - momentum) + mean * momentum)
        if running_var is not None:
            running_var._set(running_var * (1 - momentum) + var * momentum)
        inv_std = (var + eps).rsqrt()
        x_norm = (x - mean) * inv_std
        if weight is not None:
            x_norm = x_norm * weight
        if bias is not None:
            x_norm = x_norm + bias
        return x_norm
    else:
        if running_mean is None or running_var is None:
            raise RuntimeError("running_mean and running_var required in eval mode")
        return batch_norm_inference_from_tensor(x, running_mean, running_var, weight, bias, eps)


def layer_norm(x: Tensor, normalized_shape: int | Sequence[int], weight: Tensor | None = None, bias: Tensor | None = None, eps: float = 1e-5) -> Tensor:
    from torch._tensor import layer_norm_from_tensor
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    return layer_norm_from_tensor(x, normalized_shape, weight, bias, eps)


# ── Padding ───────────────────────────────────────────────────────

def pad(x: Tensor, pad: Sequence[int], mode: str = "constant", value: float = 0.0) -> Tensor:
    if mode != "constant":
        raise NotImplementedError(f"padding mode '{mode}' not yet implemented")
    if len(pad) == 0 or len(pad) > 4 or len(pad) % 2 != 0:
        raise ValueError(f"invalid pad tuple: {pad}")
    result = x
    for i in range(0, len(pad), 2):
        dim = -(i // 2 + 1)
        left = pad[i]
        right = pad[i + 1]
        if left == 0 and right == 0:
            continue
        pad_shape = list(result.shape)
        pad_shape[dim] = left
        left_pad = torch.full(pad_shape, value, dtype=result.dtype)
        pad_shape[dim] = right
        right_pad = torch.full(pad_shape, value, dtype=result.dtype)
        result = torch.cat([left_pad, result, right_pad], dim=dim)
    return result


# ── Loss functions ────────────────────────────────────────────────

def cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    log_probs = log_softmax(input, dim=-1)
    nll = nll_loss(log_probs, target, reduction="none")
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    return nll.mean()


def nll_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    from torch._tensor import nll_loss_from_tensor
    loss_per_batch = nll_loss_from_tensor(input, target)
    if reduction == "none":
        return loss_per_batch
    if reduction == "sum":
        return loss_per_batch.sum()
    return loss_per_batch.mean()


def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    diff = input - target
    loss = diff * diff
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def binary_cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    eps = 1e-12
    input_clamped = input.clamp(eps, 1.0 - eps)
    loss = -(target * input_clamped.log() + (1.0 - target) * (1.0 - input_clamped).log())
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


# ── Convolution ────────────────────────────────────

def conv2d(x: Tensor, weight: Tensor, bias: Tensor | None = None, stride: int = 1, padding: int = 0) -> Tensor:
    from torch._tensor import conv2d_from_tensors
    return conv2d_from_tensors(x, weight, bias, [stride, stride], [padding, padding], [1, 1], 1)


def max_pool2d(x: Tensor, kernel_size: int | tuple[int, int], stride: int | None = None, padding: int = 0) -> Tensor:
    from torch._tensor import max_pool2d_from_tensor
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size[0]
    if isinstance(stride, int):
        stride = (stride, stride)
    return max_pool2d_from_tensor(x, kernel_size, stride, [padding, padding], [1, 1])


def avg_pool2d(x: Tensor, kernel_size: int | tuple[int, int], stride: int | None = None, padding: int = 0) -> Tensor:
    from torch._tensor import avg_pool2d_from_tensor
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size[0]
    if isinstance(stride, int):
        stride = (stride, stride)
    return avg_pool2d_from_tensor(x, kernel_size, stride, [padding, padding], True)


# ── Internal helpers ──────────────────────────────────────────────

def _flatten(data: object) -> list[float]:
    if isinstance(data, list):
        out: list[float] = []
        for item in data:
            out.extend(_flatten(item))
        return out
    return [float(data)]


def _unflatten(shape: list[int], flat: list[float], dtype: str) -> Tensor:
    return tensor_from_data(_reshape_flat_values(flat, shape), dtype)
