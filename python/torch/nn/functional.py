from __future__ import annotations

from typing import Sequence

import torch
from torch._tensor import Tensor, tensor_from_data, _reshape_flat_values


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
    if training:
        mean = x.mean(dim=0)
        var = ((x - mean) ** 2).mean(dim=0)
        if running_mean is not None:
            flat_running = _flatten(running_mean.tolist())
            flat_mean = _flatten(mean.tolist())
            merged = [a * (1 - momentum) + b * momentum for a, b in zip(flat_running, flat_mean)]
            running_mean._set(tensor_from_data(_reshape_flat_values(merged, list(running_mean.shape)), running_mean.dtype))
        if running_var is not None:
            flat_running = _flatten(running_var.tolist())
            flat_var = _flatten(var.tolist())
            merged = [a * (1 - momentum) + b * momentum for a, b in zip(flat_running, flat_var)]
            running_var._set(tensor_from_data(_reshape_flat_values(merged, list(running_var.shape)), running_var.dtype))
    else:
        if running_mean is None or running_var is None:
            raise RuntimeError("running_mean and running_var required in eval mode")
        mean = running_mean
        var = running_var
    x_norm = (x - mean) / (var + eps).sqrt()
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    return x_norm


def layer_norm(x: Tensor, normalized_shape: int | Sequence[int], weight: Tensor | None = None, bias: Tensor | None = None, eps: float = 1e-5) -> Tensor:
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    dims = tuple(range(-len(normalized_shape), 0))
    # Reduce over all dims one by one (mean dim does not support tuple)
    mean = x
    for d in dims:
        mean = mean.mean(dim=d, keepdim=True)
    var = ((x - mean) ** 2)
    for d in dims:
        var = var.mean(dim=d, keepdim=True)
    x_norm = (x - mean) / (var + eps).sqrt()
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    return x_norm


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
    batch_size = target.shape[0]
    flat_target = _flatten(target.tolist())
    flat_input_list = _flatten(input.tolist())
    num_classes = len(flat_input_list) // batch_size
    loss_vals = []
    for i in range(batch_size):
        t = int(flat_target[i])
        idx = i * num_classes + t
        loss_vals.append(-flat_input_list[idx])
    shape_out = list(target.shape)
    if reduction == "none":
        if len(shape_out) == 0:
            return tensor_from_data(loss_vals, input.dtype)
        # reshape flat list back to target shape
        flat = loss_vals
        # use _reshape_flat_values to create nested list, then tensor_from_data
        reshaped = _reshape_flat_values(flat, shape_out)
        return tensor_from_data(reshaped, input.dtype)
    total = sum(loss_vals)
    if reduction == "sum":
        return tensor_from_data([total], input.dtype)
    return tensor_from_data([total / batch_size], input.dtype)


def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    loss = (input - target) ** 2
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


# ── Convolution (tolist-based) ────────────────────────────────────

def conv2d(x: Tensor, weight: Tensor, bias: Tensor | None = None, stride: int = 1, padding: int = 0) -> Tensor:
    if padding > 0:
        x = pad(x, [padding, padding, padding, padding])
    shape = list(x.shape)
    w_shape = list(weight.shape)
    batch_size, in_channels, in_h, in_w = shape
    out_channels, _, kernel_h, kernel_w = w_shape
    out_h = (in_h - kernel_h) // stride + 1
    out_w = (in_w - kernel_w) // stride + 1
    x_flat = _flatten(x.tolist())
    w_flat = _flatten(weight.tolist())
    b_flat = _flatten(bias.tolist()) if bias is not None else None
    out_flat = [0.0] * (batch_size * out_channels * out_h * out_w)
    for b in range(batch_size):
        for oc in range(out_channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    acc = 0.0
                    for ic in range(in_channels):
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                ih = oh * stride + kh
                                iw = ow * stride + kw
                                x_idx = ((b * in_channels + ic) * in_h + ih) * in_w + iw
                                w_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw
                                acc += x_flat[x_idx] * w_flat[w_idx]
                    if b_flat is not None:
                        acc += b_flat[oc]
                    out_idx = ((b * out_channels + oc) * out_h + oh) * out_w + ow
                    out_flat[out_idx] = acc
    return tensor_from_data(_reshape_flat_values(out_flat, [batch_size, out_channels, out_h, out_w]), x.dtype)


def max_pool2d(x: Tensor, kernel_size: int | tuple[int, int], stride: int | None = None, padding: int = 0) -> Tensor:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size[0]
    if isinstance(stride, int):
        stride = (stride, stride)
    shape = list(x.shape)
    batch_size, channels, in_h, in_w = shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    out_h = (in_h + 2 * padding - kernel_h) // stride_h + 1
    out_w = (in_w + 2 * padding - kernel_w) // stride_w + 1
    x_flat = _flatten(x.tolist())
    out_flat = [0.0] * (batch_size * channels * out_h * out_w)
    for b in range(batch_size):
        for c in range(channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    max_val = float("-inf")
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            ih = oh * stride_h + kh - padding
                            iw = ow * stride_w + kw - padding
                            if 0 <= ih < in_h and 0 <= iw < in_w:
                                val = x_flat[((b * channels + c) * in_h + ih) * in_w + iw]
                                if val > max_val:
                                    max_val = val
                    out_idx = ((b * channels + c) * out_h + oh) * out_w + ow
                    out_flat[out_idx] = max_val if max_val > float("-inf") else 0.0
    return tensor_from_data(_reshape_flat_values(out_flat, [batch_size, channels, out_h, out_w]), x.dtype)


def avg_pool2d(x: Tensor, kernel_size: int | tuple[int, int], stride: int | None = None, padding: int = 0) -> Tensor:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size[0]
    if isinstance(stride, int):
        stride = (stride, stride)
    shape = list(x.shape)
    batch_size, channels, in_h, in_w = shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    out_h = (in_h + 2 * padding - kernel_h) // stride_h + 1
    out_w = (in_w + 2 * padding - kernel_w) // stride_w + 1
    x_flat = _flatten(x.tolist())
    out_flat = [0.0] * (batch_size * channels * out_h * out_w)
    for b in range(batch_size):
        for c in range(channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    acc = 0.0
                    count = 0
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            ih = oh * stride_h + kh - padding
                            iw = ow * stride_w + kw - padding
                            if 0 <= ih < in_h and 0 <= iw < in_w:
                                acc += x_flat[((b * channels + c) * in_h + ih) * in_w + iw]
                                count += 1
                    out_idx = ((b * channels + c) * out_h + oh) * out_w + ow
                    out_flat[out_idx] = acc / count if count > 0 else 0.0
    return tensor_from_data(_reshape_flat_values(out_flat, [batch_size, channels, out_h, out_w]), x.dtype)


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
