from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ._runtime import _get_runtime, _run_js_awaitable
from .tensor_ops import _js_meta_to_tuple

if TYPE_CHECKING:
    from ._tensor import Tensor


def conv2d_from_tensors(
    input: "Tensor",
    weight: "Tensor",
    bias: "Tensor | None" = None,
    stride: Sequence[int] = (1,),
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    groups: int = 1,
) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_conv2d

    runtime = _get_runtime()
    bias_list: list[float] | None = None
    if bias is not None:
        bias_list = bias.tolist()
    meta = _run_js_awaitable(runtime.conv2d(
        input._id, weight._id, bias_list,
        [int(s) for s in stride],
        [int(p) for p in padding],
        [int(d) for d in dilation],
        int(groups),
    ))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    needs_input_grad = input._requires_grad
    needs_weight_grad = weight._requires_grad
    needs_bias_grad = bias is not None and bias._requires_grad

    if is_grad_enabled() and (needs_input_grad or needs_weight_grad or needs_bias_grad):
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=(needs_input_grad or needs_weight_grad or needs_bias_grad))
        params = (tuple(stride), tuple(padding), tuple(dilation), int(groups), bias)

        parents = [p for p in (input, weight, bias) if p is not None]
        grad_indices = [i for i, p in enumerate((input, weight, bias)) if p is not None]

        def _conv_grad_fn(g, inp=input, wt=weight, out_sh=out_shape, pr=params, gidx=grad_indices):
            all_grads = _grad_conv2d(g, inp, wt, out_sh, pr)
            return tuple(all_grads[i] for i in gidx)

        result._node = _Node(result, _conv_grad_fn, parents)
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def max_pool2d_from_tensor(
    input: "Tensor",
    kernel_size: Sequence[int],
    stride: Sequence[int] | None = None,
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    ksize = [int(k) for k in kernel_size]
    strd = [int(s) for s in (stride if stride is not None else kernel_size)]
    pad = [int(p) for p in padding]
    dil = [int(d) for d in dilation]
    meta = _run_js_awaitable(runtime.maxPool2d(input._id, ksize, strd, pad, dil))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def avg_pool2d_from_tensor(
    input: "Tensor",
    kernel_size: Sequence[int],
    stride: Sequence[int] | None = None,
    padding: Sequence[int] = (0,),
    count_include_pad: bool = True,
) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    ksize = [int(k) for k in kernel_size]
    strd = [int(s) for s in (stride if stride is not None else kernel_size)]
    pad = [int(p) for p in padding]
    meta = _run_js_awaitable(runtime.avgPool2d(input._id, ksize, strd, pad, count_include_pad))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def batch_norm_from_tensor(
    input: "Tensor",
    weight: "Tensor | None" = None,
    bias: "Tensor | None" = None,
    running_mean: "Tensor | None" = None,
    running_var: "Tensor | None" = None,
    eps: float = 1e-5,
    training: bool = False,
    momentum: float = 0.1,
) -> "Tensor":
    """Batch normalization forward pass.

    When `training=True`, computes batch statistics and updates running_mean /
    running_var in-place; the output is `(x - batch_mean) / sqrt(batch_var + eps)
    * weight + bias`. Registers an autograd node so `.backward()` flows through
    x, weight, and bias.

    When `training=False`, dispatches to the inference WGSL shader using the
    running statistics.
    """
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled
    from .autograd_rules import _grad_batch_norm
    from .grad_mode import no_grad
    runtime = _get_runtime()

    if not training:
        # Inference path: WGSL shader using running stats.
        meta = _run_js_awaitable(runtime.batchNorm(
            input._id,
            weight._id if weight is not None else None,
            bias._id if bias is not None else None,
            running_mean._id if running_mean is not None else None,
            running_var._id if running_var is not None else None,
            float(eps),
        ))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    # Training path: compute batch statistics in Python (leveraging GPU-backed
    # `mean` and arithmetic ops) and register an autograd node for backward.
    rank = len(input._shape)
    if rank < 2:
        raise RuntimeError("batch_norm expects at least 2D input (N, C, ...)")

    spatial = 1
    for s in input._shape[2:]:
        spatial *= s
    m = float(input._shape[0] * spatial)  # M = N * H * W (or N for 2D)

    with no_grad():
        # batch mean, shape (C,): average over dims 0, 2, 3 (or 0 for 2D).
        mean = input
        for d in range(rank):
            if d == 1:
                continue
            mean = mean.mean(dim=d, keepdim=True)
        # Squeeze the non-channel dims so mean has shape (C,).
        for d in range(rank - 1, -1, -1):
            if d == 1:
                continue
            mean = mean.squeeze(d) if mean._shape[d] == 1 else mean

        # Centered input and its variance.
        # Broadcast mean (C,) over the leading N and trailing spatial dims:
        # add a view of mean with shape (1, C, 1, 1) ... or (1, C).
        mean_view = mean
        for d in range(rank):
            if d == 1:
                continue
            mean_view = mean_view.unsqueeze(d) if mean._shape[d] == 1 else mean_view
        # `unsqueeze` after squeezing leaves mean_view with shape (1, C, 1, 1) for 4D
        # or (1, C) for 2D — matching the broadcast pattern.

        x_centered = input.sub(mean_view)
        # variance = mean(x_centered^2)
        var = x_centered.pow(2)
        for d in range(rank):
            if d == 1:
                continue
            var = var.mean(dim=d, keepdim=True)
        for d in range(rank - 1, -1, -1):
            if d == 1:
                continue
            var = var.squeeze(d) if var._shape[d] == 1 else var

        inv_std = (var + float(eps)).rsqrt()

        # x_hat: normalized but not affine, same shape as input.
        inv_std_view = inv_std
        for d in range(rank):
            if d == 1:
                continue
            inv_std_view = inv_std_view.unsqueeze(d) if inv_std._shape[d] == 1 else inv_std_view

        x_hat = x_centered.mul(inv_std_view)
        # Apply affine (optional).
        if weight is not None:
            x_hat_aff = x_hat.mul(weight)
        else:
            x_hat_aff = x_hat
        if bias is not None:
            output = x_hat_aff.add(bias)
        else:
            output = x_hat_aff

        # Update running stats in-place (outside autograd graph).
        if running_mean is not None:
            new_rm = mean.mul(float(momentum)).add(running_mean.mul(1.0 - float(momentum)))
            running_mean._set(new_rm)
        if running_var is not None:
            # PyTorch's running_var stores the *unbiased* variance estimate
            # (population variance with Bessel's correction: var * M / (M-1)).
            # This matches what real PyTorch's BN layer expects when restored
            # from state_dict.
            unbiased_var = var.mul(m / (m - 1.0))
            new_rv = unbiased_var.mul(float(momentum)).add(running_var.mul(1.0 - float(momentum)))
            running_var._set(new_rv)

    # The output tensor inherits requires_grad based on its inputs.
    any_requires_grad = (
        input._requires_grad
        or (weight is not None and weight._requires_grad)
        or (bias is not None and bias._requires_grad)
    )
    if is_grad_enabled() and any_requires_grad:
        result = output
        result._requires_grad = True
        # Save the tensors needed for backward. We attach them to the closure.
        saved_x = input
        saved_weight = weight
        saved_bias = bias
        saved_x_hat = x_hat
        saved_inv_std = inv_std

        def grad_fn(grad_output: "Tensor"):
            return _grad_batch_norm(
                grad_output,
                saved_x,
                saved_weight,
                saved_bias,
                saved_x_hat,
                saved_inv_std,
                int(m),
            )

        # parents must be in the same order as grad_fn returns:
        # (grad_x, grad_weight, grad_bias). Skip None for optional params.
        parents = [input]
        if weight is not None:
            parents.append(weight)
        if bias is not None:
            parents.append(bias)
        result._node = _Node(result, grad_fn, parents)
        return result
    return output


def nll_loss_from_tensor(
    input: "Tensor",
    target: "Tensor",
    reduction: str = "none",
) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_nll_loss

    runtime = _get_runtime()
    if reduction in ("sum", "mean"):
        meta = _run_js_awaitable(runtime.nllLossReduced(input._id, target._id, reduction))
    else:
        meta = _run_js_awaitable(runtime.nllLoss(input._id, target._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and input._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_nll_loss(g, input, target),), [input])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def cross_entropy_from_tensor(
    input: "Tensor",
    target: "Tensor",
    reduction: str = "mean",
) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled
    from .autograd_rules import _grad_cross_entropy_fused

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.crossEntropy(input._id, target._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    loss_per_batch = Tensor(tensor_id, out_shape, out_dtype)

    if reduction == "none":
        result = loss_per_batch
    elif reduction == "sum":
        result = loss_per_batch.sum()
    else:
        result = loss_per_batch.mean()

    if is_grad_enabled() and input._requires_grad:
        result._requires_grad = True
        result._node = _Node(result, lambda g: (_grad_cross_entropy_fused(g, input, target, reduction),), [input])
    return result


def batch_norm_inference_from_tensor(
    input: "Tensor",
    running_mean: "Tensor",
    running_var: "Tensor",
    weight: "Tensor | None" = None,
    bias: "Tensor | None" = None,
    eps: float = 1e-5,
) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.batchNorm(
        input._id,
        weight._id if weight is not None else None,
        bias._id if bias is not None else None,
        running_mean._id,
        running_var._id,
        float(eps),
    ))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def layer_norm_from_tensor(
    input: "Tensor",
    normalized_shape: Sequence[int],
    weight: "Tensor | None" = None,
    bias: "Tensor | None" = None,
    eps: float = 1e-5,
) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.layerNorm(
        input._id,
        [int(s) for s in normalized_shape],
        weight._id if weight is not None else None,
        bias._id if bias is not None else None,
        float(eps),
    ))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)
