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
    from torch.autograd import _Node, is_grad_enabled
    from torch.autograd_rules import _grad_dropout
    from torch.grad_mode import no_grad
    with no_grad():
        mask_f = (_torch.rand(list(x.shape), dtype=x.dtype) > float(p)).to(x.dtype)
    scale = 1.0 / (1.0 - float(p)) if p < 1.0 else 0.0
    output = x.mul(mask_f).mul(scale)
    if is_grad_enabled() and x._requires_grad:
        output._requires_grad = True
        saved_x = x
        saved_mask = mask_f
        saved_p = float(p)
        def grad_fn(grad_output: Tensor):
            return (_grad_dropout(grad_output, saved_x, saved_mask, saved_p),)
        output._node = _Node(output, grad_fn, [x])
    return output


def dropout2d(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training or p == 0.0:
        return x
    import torch as _torch
    from torch.autograd import _Node, is_grad_enabled
    from torch.autograd_rules import _grad_dropout
    from torch.grad_mode import no_grad
    with no_grad():
        shape = list(x.shape)
        mask_shape = [shape[0], shape[1]] + [1] * (len(shape) - 2)
        mask_f = (_torch.rand(mask_shape, dtype=x.dtype) > float(p)).to(x.dtype)
    scale = 1.0 / (1.0 - float(p)) if p < 1.0 else 0.0
    # Broadcast mask over spatial dims.
    output = x.mul(mask_f).mul(scale)
    if is_grad_enabled() and x._requires_grad:
        output._requires_grad = True
        saved_x = x
        saved_mask = mask_f
        saved_p = float(p)
        def grad_fn(grad_output: Tensor):
            return (_grad_dropout(grad_output, saved_x, saved_mask, saved_p),)
        output._node = _Node(output, grad_fn, [x])
    return output


# ── Linear ────────────────────────────────────────────────────────

def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    result = x.matmul(weight.T)
    if bias is not None:
        # Broadcast the (out_features,) bias over the leading batch dims.
        bias_view = bias.reshape([1] * (result.ndim - 1) + [result.shape[-1]])
        result = result + bias_view
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
        bias_view = bias.reshape([1] * (result.ndim - 1) + [result.shape[-1]])
        result = result + bias_view

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
    from torch._tensor import batch_norm_from_tensor
    return batch_norm_from_tensor(
        x,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        eps=eps,
        training=training,
        momentum=momentum,
    )


def layer_norm(x: Tensor, normalized_shape: int | Sequence[int], weight: Tensor | None = None, bias: Tensor | None = None, eps: float = 1e-5) -> Tensor:
    from torch._tensor import layer_norm_from_tensor
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    return layer_norm_from_tensor(x, normalized_shape, weight, bias, eps)


def group_norm(x: Tensor, num_groups: int, weight: Tensor | None = None, bias: Tensor | None = None, eps: float = 1e-5) -> Tensor:
    """Apply Group Normalization (Wu & He, 2018).

    Splits the channel dim of `x` (shape `(N, C, *)`) into `num_groups`
    groups, normalizes within each group, then applies a per-channel affine.
    """
    from torch._tensor import group_norm_from_tensor
    return group_norm_from_tensor(x, int(num_groups), weight, bias, eps)


# ── Padding ───────────────────────────────────────────────────────

def pad(x: Tensor, pad: Sequence[int], mode: str = "constant", value: float = 0.0) -> Tensor:
    if len(pad) == 0 or len(pad) > 8 or len(pad) % 2 != 0:
        raise ValueError(f"invalid pad tuple: {pad}")
    if mode in ("constant", "replicate", "reflect", "circular"):
        gpu = _gpu_pad(x, pad, mode, value)
        if gpu is not None:
            return gpu
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


def _gpu_pad(x: Tensor, pad: Sequence[int], mode: str, value: float = 0.0) -> "Tensor | None":
    """Route standard 1D/2D padding through the dedicated GPU shaders.

    Returns the GPU-padded tensor for padding the last 1 or 2 spatial dims,
    or ``None`` to signal the caller to fall back to the per-dimension
    emulation (e.g. for 3D padding or non-trailing dims).
    """
    if len(pad) not in (2, 4):
        return None
    from torch._tensor_runtime_bridge import (
        circular_pad_from_tensor,
        constant_pad_from_tensor,
        reflection_pad_from_tensor,
        replication_pad_from_tensor,
    )

    left, right = pad[0], pad[1]
    top = bottom = 0
    if len(pad) == 4:
        top, bottom = pad[2], pad[3]
    if mode == "replicate":
        return replication_pad_from_tensor(x, left, right, top, bottom)
    if mode == "reflect":
        return reflection_pad_from_tensor(x, left, right, top, bottom)
    if mode == "circular":
        return circular_pad_from_tensor(x, left, right, top, bottom)
    if mode == "constant":
        return constant_pad_from_tensor(x, left, right, top, bottom, value)
    return None


def _pad_reflect(x: Tensor, dim: int, left: int, right: int) -> Tensor:
    """Reflect padding: pads with reflection of tensor at boundaries.

    Mirrors about the boundary without repeating the edge value.
    E.g. input [1,2,3,4] with left=2, right=2 -> [3,2,1,2,3,4,3,2].
    """
    dim_size = x.shape[dim]
    parts = []
    if left > 0:
        left_slice = x.narrow(dim, 1, min(left, dim_size - 1)).flip([dim])
        if left_slice.shape[dim] < left:
            extra = x.narrow(dim, 0, left - left_slice.shape[dim]).flip([dim])
            left_slice = torch.cat([extra, left_slice], dim=dim)
        parts.append(left_slice)
    parts.append(x)
    if right > 0:
        right_slice = x.narrow(dim, max(0, dim_size - 1 - right), min(right, dim_size - 1)).flip([dim])
        if right_slice.shape[dim] < right:
            remaining = right - right_slice.shape[dim]
            extra = x.narrow(dim, dim_size - remaining, remaining).flip([dim])
            right_slice = torch.cat([right_slice, extra], dim=dim)
        parts.append(right_slice)
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=dim)


def _pad_replicate(x: Tensor, dim: int, left: int, right: int) -> Tensor:
    """Replicate padding: pads with replication of edge values."""
    parts = []
    if left > 0:
        # Repeat the first element `left` times along `dim` via narrow+cat loop.
        first = x.narrow(dim, 0, 1)
        left_part = first
        while left_part.shape[dim] < left:
            left_part = torch.cat([left_part, first], dim=dim)
        if left_part.shape[dim] > left:
            left_part = left_part.narrow(dim, 0, left)
        parts.append(left_part)
    parts.append(x)
    if right > 0:
        last = x.narrow(dim, x.shape[dim] - 1, 1)
        right_part = last
        while right_part.shape[dim] < right:
            right_part = torch.cat([right_part, last], dim=dim)
        if right_part.shape[dim] > right:
            right_part = right_part.narrow(dim, 0, right)
        parts.append(right_part)
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=dim)


def _pad_circular(x: Tensor, dim: int, left: int, right: int) -> Tensor:
    """Circular padding: pads with circular repetition of tensor."""
    parts = []
    size = x.shape[dim]
    if left > 0:
        # Take the last `left` elements.
        left_part = x.narrow(dim, size - left, left)
        parts.append(left_part)
    parts.append(x)
    if right > 0:
        # Take the first `right` elements.
        right_part = x.narrow(dim, 0, right)
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
    # celu(x) = x if x >= 0, else alpha * (exp(x/alpha) - 1)
    # Using where matches the elu pattern; gradient at x=0 is alpha*exp(0)=1.
    return torch.where(x > 0.0, x, alpha * (x.div(alpha).exp().sub(1.0)))


def softplus(x: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    # Numerical stability: when input * beta > threshold, softplus(x) ≈ x.
    if beta == 1.0 and threshold == 20.0:
        return x.softplus()
    scaled = x * beta
    use_linear = scaled > threshold
    exact = (scaled.exp().add(1.0).log()) / beta
    return torch.where(use_linear, x, exact)


def mish(x: Tensor) -> Tensor:
    return x.mish()


def hardswish(x: Tensor) -> Tensor:
    return x.hardswish()


def hardsigmoid(x: Tensor) -> Tensor:
    return x.hardsigmoid()


def softsign(x: Tensor) -> Tensor:
    return x.softsign()


def tanhshrink(x: Tensor) -> Tensor:
    return x.tanhshrink()


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
    from torch._tensor import cross_entropy_from_tensor
    return cross_entropy_from_tensor(input, target, reduction)


def nll_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    from torch._tensor import nll_loss_from_tensor
    if reduction == "none":
        return nll_loss_from_tensor(input, target, "none")
    if reduction == "sum":
        return nll_loss_from_tensor(input, target, "sum")
    return nll_loss_from_tensor(input, target, "mean")


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
    max_val = torch.clamp(-input, 0.0, float("inf"))
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


def huber_loss(input: Tensor, target: Tensor, reduction: str = "mean", delta: float = 1.0) -> Tensor:
    """Huber loss: smooth L1 with `delta` transition. Same as smooth_l1_loss
    with `beta=delta`."""
    return smooth_l1_loss(input, target, reduction=reduction, beta=delta)


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
        return _interpolate_gpu(x, out_h, out_w, "nearest", align_corners)
    elif mode == "bilinear":
        return _interpolate_gpu(x, out_h, out_w, "bilinear", align_corners)
    elif mode == "bicubic":
        # No dedicated bicubic shader yet; fall back to bilinear (existing behavior).
        return _interpolate_bilinear(x, out_h, out_w, align_corners)
    else:
        raise ValueError(f"Unknown interpolation mode: {mode}")


def _interpolate_gpu(x: Tensor, out_h: int, out_w: int, mode: str, align_corners: bool | None) -> Tensor:
    """Route nearest/bilinear upsampling through the dedicated GPU shader.

    Falls back to the Python emulation for non-4D inputs.
    """
    if x.ndim == 4:
        from torch._tensor_runtime_bridge import upsample2d_from_tensor

        return upsample2d_from_tensor(x, out_h, out_w, mode, bool(align_corners))
    if mode == "nearest":
        return _interpolate_nearest(x, out_h, out_w)
    return _interpolate_bilinear(x, out_h, out_w, align_corners)


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
    """Bilinear interpolation using tensor operations (no tolist)."""
    import torch
    in_h, in_w = x.shape[2], x.shape[3]
    batch_size, channels = x.shape[0], x.shape[1]

    if align_corners:
        h_scale = (in_h - 1) / (out_h - 1) if out_h > 1 else 0.0
        w_scale = (in_w - 1) / (out_w - 1) if out_w > 1 else 0.0
    else:
        h_scale = in_h / out_h
        w_scale = in_w / out_w

    # Build coordinate grids as tensors.
    h_grid = torch.arange(0, out_h, 1, dtype=x.dtype).mul(h_scale)
    w_grid = torch.arange(0, out_w, 1, dtype=x.dtype).mul(w_scale)

    # Floor coordinates (clamped so ih1/iw1 is always valid).
    ih0 = h_grid.floor().to(dtype="int64").clamp(0, in_h - 2)
    ih1 = ih0.add(1)
    iw0 = w_grid.floor().to(dtype="int64").clamp(0, in_w - 2)
    iw1 = iw0.add(1)

    # Fractional parts.
    di = h_grid.sub(ih0.to(dtype=x.dtype)).reshape([1, 1, out_h, 1])
    dj = w_grid.sub(iw0.to(dtype=x.dtype)).reshape([1, 1, 1, out_w])

    # Flatten x to [B, C, H*W] for gather.
    x_flat = x.reshape(batch_size, channels, -1)

    # Flat indices: [oh, ow] -> ih * W + iw, broadcast to [B, C, out_h, out_w].
    def _gather_weights(hi, wi):
        flat_idx = (
            hi.reshape([1, 1, out_h, 1]).mul(in_w)
            .add(wi.reshape([1, 1, 1, out_w]))
            .expand([batch_size, channels, out_h, out_w])
            .reshape(batch_size, channels, -1)
        )
        return x_flat.gather(2, flat_idx).reshape(batch_size, channels, out_h, out_w)

    v00 = _gather_weights(ih0, iw0)
    v01 = _gather_weights(ih0, iw1)
    v10 = _gather_weights(ih1, iw0)
    v11 = _gather_weights(ih1, iw1)

    out = (
        v00.mul((1 - di) * (1 - dj))
        .add(v01.mul((1 - di) * dj))
        .add(v10.mul(di * (1 - dj)))
        .add(v11.mul(di * dj))
    )
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


def embedding(input: Tensor, weight: Tensor, padding_idx: int = -1) -> Tensor:
    from torch.tensor_nn_ops import embedding_from_tensor
    return embedding_from_tensor(weight, input, padding_idx)


# ── Additional activations ────────────────────────────────────────

def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def logsigmoid(x: Tensor) -> Tensor:
    return x.neg().softplus().neg()


def softmin(x: Tensor, dim: int = -1) -> Tensor:
    return x.neg().softmax(dim)


def hardtanh(x: Tensor, min_val: float = -1.0, max_val: float = 1.0) -> Tensor:
    return x.clamp(min=min_val, max=max_val)


def relu6(x: Tensor) -> Tensor:
    return x.clamp(min=0.0, max=6.0)


def hardshrink(x: Tensor, lambd: float = 0.5) -> Tensor:
    return torch.where(x.abs() > lambd, x, 0.0)


def softshrink(x: Tensor, lambd: float = 0.5) -> Tensor:
    pos = torch.where(x > lambd, x - lambd, 0.0)
    neg = torch.where(x < -lambd, x + lambd, 0.0)
    return pos + neg


def threshold(x: Tensor, threshold: float, value: float) -> Tensor:
    return torch.where(x > threshold, x, value)


def selu(x: Tensor) -> Tensor:
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return torch.where(x > 0.0, x, alpha * (x.exp() - 1.0)) * scale


def gumbel_softmax(logits: Tensor, tau: float = 1.0, hard: bool = False, dim: int = -1) -> Tensor:
    import torch as _torch
    u = _torch.rand(list(logits.shape), dtype=logits.dtype)
    gumbel = u.log().neg().log().neg()
    y = ((logits + gumbel) / tau).softmax(dim)
    if hard:
        index = y.argmax(dim=dim, keepdim=True)
        y_hard = _torch.scatter(_torch.zeros(list(y.shape), dtype=y.dtype), dim, index, 1.0)
        return (y_hard - y).add(y)
    return y


# ── Distances ─────────────────────────────────────────────────────

def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    dot = (x1 * x2).sum(dim=dim)
    n1 = x1.mul(x1).sum(dim=dim).sqrt()
    n2 = x2.mul(x2).sum(dim=dim).sqrt()
    return dot.div((n1 * n2).clamp(min=eps))


def pairwise_distance(x1: Tensor, x2: Tensor, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> Tensor:
    diff = (x1 - x2).abs().add(eps)
    return diff.pow(p).sum(dim=-1, keepdim=keepdim).pow(1.0 / p)


def pdist(input: Tensor, p: float = 2.0) -> Tensor:
    return torch.pdist(input, p=p)


# ── Normalization ─────────────────────────────────────────────────

def rms_norm(x: Tensor, normalized_shape, weight: Tensor | None = None, eps: float | None = None) -> Tensor:
    n = len(normalized_shape)
    ndim = len(x.shape)
    ms = x.mul(x)
    for i in range(n):
        ms = ms.mean(dim=ndim - 1 - i, keepdim=True)
    e = 1e-6 if eps is None else float(eps)
    out = x.mul(ms.add(e).rsqrt())
    if weight is not None:
        out = out.mul(weight)
    return out


# ── Scaled dot-product attention ──────────────────────────────────

def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> Tensor:
    import math
    import torch as _torch
    e = query.shape[-1]
    scale_factor = (1.0 / math.sqrt(e)) if scale is None else float(scale)
    scores = query.matmul(key.transpose(-2, -1)).mul(scale_factor)
    if is_causal:
        lq = query.shape[-2]
        lk = key.shape[-2]
        mask = _torch.ones([lq, lk], dtype="float32").tril()
        neg = (1.0 - mask).mul(-1e9)
        scores = scores.add(neg)
    if attn_mask is not None:
        if attn_mask._dtype == "bool":
            scores = scores.add(attn_mask.logical_not().to("float32").mul(-1e9))
        else:
            scores = scores.add(attn_mask)
    attn = scores.softmax(-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p, training=True)
    return attn.matmul(value)


# ── Pixel shuffle ─────────────────────────────────────────────────

def pixel_shuffle(input: Tensor, upscale_factor: int) -> Tensor:
    r = upscale_factor
    *batch, c, h, w = list(input.shape)
    oc = c // (r * r)
    x = input.reshape(batch + [oc, r, r, h, w])
    nd = len(batch)
    perm = list(range(nd)) + [nd, nd + 3, nd + 1, nd + 4, nd + 2]
    x = x.permute(perm)
    return x.reshape(batch + [oc, h * r, w * r])


def pixel_unshuffle(input: Tensor, downscale_factor: int) -> Tensor:
    r = downscale_factor
    *batch, c, h, w = list(input.shape)
    oh, ow = h // r, w // r
    x = input.reshape(batch + [c, oh, r, ow, r])
    nd = len(batch)
    perm = list(range(nd)) + [nd, nd + 2, nd + 4, nd + 1, nd + 3]
    x = x.permute(perm)
    return x.reshape(batch + [c * r * r, oh, ow])


# ── Additional losses ─────────────────────────────────────────────

def kl_div(input: Tensor, target: Tensor, reduction: str = "mean", log_target: bool = False) -> Tensor:
    if log_target:
        loss = target.exp().mul(target - input)
    else:
        loss = target.mul(target.add(1e-12).log() - input)
    if reduction == "batchmean":
        return loss.sum().div(input.shape[0])
    return _reduce(loss, reduction)


def soft_margin_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    loss = target.neg().mul(input).softplus()
    return _reduce(loss, reduction)


def hinge_embedding_loss(input: Tensor, target: Tensor, margin: float = 1.0, reduction: str = "mean") -> Tensor:
    pos = input
    neg = (margin - input).clamp(min=0.0)
    loss = torch.where(target > 0.0, pos, neg)
    return _reduce(loss, reduction)


def margin_ranking_loss(input1: Tensor, input2: Tensor, target: Tensor, margin: float = 0.0, reduction: str = "mean") -> Tensor:
    loss = (target.neg().mul(input1 - input2) + margin).clamp(min=0.0)
    return _reduce(loss, reduction)


def cosine_embedding_loss(input1: Tensor, input2: Tensor, target: Tensor, margin: float = 0.0, reduction: str = "mean") -> Tensor:
    cos = cosine_similarity(input1, input2, dim=1)
    pos = 1.0 - cos
    neg = (cos - margin).clamp(min=0.0)
    loss = torch.where(target > 0.0, pos, neg)
    return _reduce(loss, reduction)


def poisson_nll_loss(input: Tensor, target: Tensor, log_input: bool = True, full: bool = False, eps: float = 1e-8, reduction: str = "mean") -> Tensor:
    if log_input:
        loss = input.exp() - target.mul(input)
    else:
        loss = input - target.mul(input.add(eps).log())
    return _reduce(loss, reduction)


def triplet_margin_loss(anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = 1.0, p: float = 2.0, eps: float = 1e-6, reduction: str = "mean") -> Tensor:
    dpos = pairwise_distance(anchor, positive, p=p, eps=eps)
    dneg = pairwise_distance(anchor, negative, p=p, eps=eps)
    loss = (dpos - dneg + margin).clamp(min=0.0)
    return _reduce(loss, reduction)


# ── Aliases / additional pooling ──────────────────────────────────

def adaptive_avg_pool1d(x: Tensor, output_size: int) -> Tensor:
    L = x.shape[-1]
    if L == output_size:
        return x
    if output_size == 1:
        return x.mean(dim=-1, keepdim=True)
    stride = L // output_size
    kernel = L - (output_size - 1) * stride
    return avg_pool1d(x, kernel, stride, 0)


def upsample(input: Tensor, size=None, scale_factor=None, mode: str = "nearest", align_corners=None) -> Tensor:
    return interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


def upsample_nearest(input: Tensor, size=None, scale_factor=None) -> Tensor:
    return interpolate(input, size=size, scale_factor=scale_factor, mode="nearest")


def upsample_bilinear(input: Tensor, size=None, scale_factor=None) -> Tensor:
    return interpolate(input, size=size, scale_factor=scale_factor, mode="bilinear", align_corners=True)
