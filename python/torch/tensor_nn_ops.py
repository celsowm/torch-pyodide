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
) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
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


def nll_loss_from_tensor(
    input: "Tensor",
    target: "Tensor",
) -> "Tensor":
    from ._tensor import Tensor
    from .autograd import _Node, is_grad_enabled, _grad_nll_loss

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.nllLoss(input._id, target._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and input._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_nll_loss(g, input, target),), [input])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


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
