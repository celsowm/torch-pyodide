from __future__ import annotations

from typing import Sequence

import torch
from torch._tensor import Tensor
from torch.tensor_factories_ops import tensor_from_data


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
    import torch as _torch
    mask = _torch.rand(list(x.shape), dtype=x.dtype) > p
    return x.mul(mask).div(1.0 - p)


def dropout2d(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training or p == 0.0:
        return x
    import torch as _torch
    shape = list(x.shape)
    mask_shape = [shape[0], shape[1]] + [1] * (len(shape) - 2)
    mask = (_torch.rand(mask_shape, dtype=x.dtype) > p).to(x.dtype)
    return x.mul(mask).div(1.0 - p)


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
    if len(pad) == 0 or len(pad) > 8 or len(pad) % 2 != 0:
        raise ValueError(f"invalid pad tuple: {pad}")
    result = x
    for i in range(0, len(pad), 2):
        dim = -(i // 2 + 1)
        left = pad[i]
        right = pad[i + 1]
        if left == 0 and right == 0:
            continue
        if mode == "constant":
            pad_shape = list(result.shape)
            pad_shape[dim] = left
            left_pad = torch.full(pad_shape, value, dtype=result.dtype)
            pad_shape[dim] = right
            right_pad = torch.full(pad_shape, value, dtype=result.dtype)
            result = torch.cat([left_pad, result, right_pad], dim=dim)
        elif mode == "reflect":
            result = _pad_reflect(result, dim, left, right)
        elif mode == "replicate":
            result = _pad_replicate(result, dim, left, right)
        elif mode == "circular":
            result = _pad_circular(result, dim, left, right)
        else:
            raise NotImplementedError(f"padding mode '{mode}' not yet implemented")
    return result


def _pad_reflect(x: Tensor, dim: int, left: int, right: int) -> Tensor:
    """Reflect padding: pads with reflection of tensor at boundaries."""
    dim_size = x.shape[dim]
    parts = []
    # Left padding: reflect from left boundary
    if left > 0:
        # indices: 1, 2, ..., left (modulated to reflect)
        indices = []
        for i in range(left, 0, -1):
            idx = i % (2 * dim_size)
            if idx >= dim_size:
                idx = 2 * dim_size - 1 - idx
            indices.append(idx)
        left_part = x.index_select(dim, torch.tensor(indices, dtype=torch.int32))
        parts.append(left_part)
    parts.append(x)
    # Right padding: reflect from right boundary
    if right > 0:
        indices = []
        for i in range(right):
            idx = (dim_size - 2 - i) % (2 * dim_size)
            if idx < 0:
                idx = -1 - idx
            if idx >= dim_size:
                idx = 2 * dim_size - 1 - idx
            indices.append(idx)
        right_part = x.index_select(dim, torch.tensor(indices, dtype=torch.int32))
        parts.append(right_part)
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=dim)


def _pad_replicate(x: Tensor, dim: int, left: int, right: int) -> Tensor:
    """Replicate padding: pads with replication of edge values."""
    parts = []
    if left > 0:
        left_idx = [0] * left
        left_part = x.index_select(dim, torch.tensor(left_idx, dtype=torch.int32))
        parts.append(left_part)
    parts.append(x)
    if right > 0:
        size = x.shape[dim]
        right_idx = [size - 1] * right
        right_part = x.index_select(dim, torch.tensor(right_idx, dtype=torch.int32))
        parts.append(right_part)
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=dim)


def _pad_circular(x: Tensor, dim: int, left: int, right: int) -> Tensor:
    """Circular padding: pads with circular repetition of tensor."""
    parts = []
    size = x.shape[dim]
    if left > 0:
        left_indices = list(range(size - left, size))
        left_part = x.index_select(dim, torch.tensor(left_indices, dtype=torch.int32))
        parts.append(left_part)
    parts.append(x)
    if right > 0:
        right_indices = list(range(right))
        right_part = x.index_select(dim, torch.tensor(right_indices, dtype=torch.int32))
        parts.append(right_part)
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=dim)


# ── Activation functions (extended) ───────────────────────────────

def prelu(x: Tensor, weight: Tensor) -> Tensor:
    return torch.where(x > 0.0, x, x * weight)


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    return torch.where(x > 0.0, x, alpha * (x.exp() - 1.0))


def celu(x: Tensor, alpha: float = 1.0) -> Tensor:
    return torch.max(x, torch.zeros_like(x)) + alpha * (torch.min(x, torch.zeros_like(x)) / alpha).exp().sub(1.0).mul(alpha)


def rrelu(x: Tensor, lower: float = 0.125, upper: float = 0.3333333333333333, training: bool = True) -> Tensor:
    import random as _random
    if training:
        a = _random.uniform(lower, upper)
    else:
        a = (lower + upper) / 2.0
    return torch.where(x > 0.0, x, x * a)


def glu(x: Tensor, dim: int = -1) -> Tensor:
    size = x.shape[dim] // 2
    a, b = x.split(size, dim=dim)
    return a * torch.sigmoid(b)


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


def binary_cross_entropy_with_logits(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    # sigmoid + BCE fused
    max_val = torch.clamp(-input, 0.0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def l1_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    loss = (input - target).abs()
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def smooth_l1_loss(input: Tensor, target: Tensor, reduction: str = "mean", beta: float = 1.0) -> Tensor:
    diff = input - target
    abs_diff = diff.abs()
    loss = torch.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
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


# ── Interpolate / Upsample ────────────────────────────────────────

def interpolate(
    x: Tensor,
    size: int | tuple[int, ...] | None = None,
    scale_factor: float | tuple[float, ...] | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
) -> Tensor:
    """Upsamples or downsamples input using nearest, bilinear, or bicubic interpolation."""
    import torch
    if size is None and scale_factor is None:
        raise ValueError("either size or scale_factor must be specified")
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor can be specified")

    if isinstance(size, int):
        size = (size, size)
    if isinstance(scale_factor, (int, float)):
        scale_factor = (float(scale_factor), float(scale_factor))

    if size is None and scale_factor is not None:
        size = tuple(int(s * f) for s, f in zip(x.shape[2:], scale_factor))

    out_h, out_w = size
    in_h, in_w = x.shape[2], x.shape[3]

    if mode == "nearest":
        return _interpolate_nearest(x, out_h, out_w)
    elif mode == "bilinear":
        return _interpolate_bilinear(x, out_h, out_w, align_corners)
    elif mode == "bicubic":
        return _interpolate_bicubic(x, out_h, out_w, align_corners)
    else:
        raise ValueError(f"Unknown interpolation mode: {mode}")


def _interpolate_nearest(x: Tensor, out_h: int, out_w: int) -> Tensor:
    """Nearest neighbor interpolation via index_select."""
    import torch
    in_h, in_w = x.shape[2], x.shape[3]
    # Build row indices
    h_indices = [int(i * in_h / out_h) for i in range(out_h)]
    w_indices = [int(j * in_w / out_w) for j in range(out_w)]

    # Interpolate H dimension
    result = None
    for hi, idx in enumerate(h_indices):
        row = x.select(2, idx)  # [N, C, W]
        # Now interpolate W
        expanded_rows = []
        for wj, widx in enumerate(w_indices):
            col = row.select(2, widx)  # [N, C]
            expanded_rows.append(col.unsqueeze(2))  # [N, C, 1]
        h_row = torch.cat(expanded_rows, dim=2)  # [N, C, W']
        if result is None:
            result = h_row.unsqueeze(2)
        else:
            result = torch.cat([result, h_row.unsqueeze(2)], dim=2)
    return result


def _interpolate_bilinear(x: Tensor, out_h: int, out_w: int, align_corners: bool | None) -> Tensor:
    """Bilinear interpolation using tensor operations (efficient)."""
    import torch
    in_h, in_w = x.shape[2], x.shape[3]
    batch_size, channels = x.shape[0], x.shape[1]

    if align_corners:
        h_scale = (in_h - 1) / (out_h - 1) if out_h > 1 else 0
        w_scale = (in_w - 1) / (out_w - 1) if out_w > 1 else 0
    else:
        h_scale = in_h / out_h
        w_scale = in_w / out_w

    # Create coordinate grids
    h_coords = [(i * h_scale) for i in range(out_h)]
    w_coords = [(j * w_scale) for j in range(out_w)]

    # For each output coordinate, interpolate from 4 nearest input pixels
    out = torch.zeros((batch_size, channels, out_h, out_w), dtype=x.dtype)

    for oh in range(out_h):
        ih_f = h_coords[oh]
        ih0 = int(ih_f)
        ih1 = min(in_h - 1, ih0 + 1)
        di = ih_f - ih0

        for ow in range(out_w):
            iw_f = w_coords[ow]
            iw0 = int(iw_f)
            iw1 = min(in_w - 1, iw0 + 1)
            dj = iw_f - iw0

            # Bilinear: v = (1-di)(1-dj)*v00 + (1-di)*dj*v01 + di*(1-dj)*v10 + di*dj*v11
            v00 = x.select(2, ih0).select(2, iw0)
            v01 = x.select(2, ih0).select(2, iw1)
            v10 = x.select(2, ih1).select(2, iw0)
            v11 = x.select(2, ih1).select(2, iw1)

            v = (
                v00.mul((1 - di) * (1 - dj))
                .add(v01.mul((1 - di) * dj))
                .add(v10.mul(di * (1 - dj)))
                .add(v11.mul(di * dj))
            )

            # Use index assignment
            out_data = out.tolist()
            for n in range(batch_size):
                for c in range(channels):
                    out_data[n][c][oh][ow] = v.tolist()[n][c]
            out = torch.tensor(out_data, dtype=x.dtype)
    return out


def _interpolate_bicubic(x: Tensor, out_h: int, out_w: int, align_corners: bool | None) -> Tensor:
    """Bicubic interpolation (simplified - falls back to bilinear)."""
    return _interpolate_bilinear(x, out_h, out_w, align_corners)


# ── normalize / one_hot ──────────────────────────────────────────

def normalize(x: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> Tensor:
    """L2 normalization along a dimension."""
    norm = x.norm(p=p, dim=dim, keepdim=True)
    return x.div(norm.clamp(min=eps))


def one_hot(tensor: Tensor, num_classes: int | None = None) -> Tensor:
    import torch as _torch
    if num_classes is None:
        num_classes = int(tensor.max().item()) + 1
    classes = _torch.arange(num_classes, dtype=tensor.dtype)
    expanded = tensor.unsqueeze(-1).to(tensor.dtype)
    result = (expanded == classes).to(tensor.dtype)
    return result


# ── Pooling 1D ───────────────────────────────────────────────────

def max_pool1d(x: Tensor, kernel_size: int, stride: int | None = None, padding: int = 0) -> Tensor:
    """1D max pooling. Adds dummy H dim and uses max_pool2d."""
    if stride is None:
        stride = kernel_size
    # [N, C, L] -> [N, C, 1, L]
    x_2d = x.unsqueeze(2)
    result = max_pool2d(x_2d, (1, kernel_size), (1, stride), (0, padding))
    return result.squeeze(2)


def avg_pool1d(x: Tensor, kernel_size: int, stride: int | None = None, padding: int = 0) -> Tensor:
    """1D average pooling. Adds dummy H dim and uses avg_pool2d."""
    if stride is None:
        stride = kernel_size
    x_2d = x.unsqueeze(2)
    result = avg_pool2d(x_2d, (1, kernel_size), (1, stride), padding)
    return result.squeeze(2)



