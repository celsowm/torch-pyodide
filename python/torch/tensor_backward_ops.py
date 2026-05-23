from __future__ import annotations

from typing import TYPE_CHECKING

from ._runtime import _get_runtime, _run_js_awaitable
from .tensor_ops import _js_meta_to_tuple

if TYPE_CHECKING:
    from ._tensor import Tensor


def conv2d_input_backward_from_tensors(
    grad_output: "Tensor",
    weight: "Tensor",
    input_shape: tuple[int, ...],
    grad_output_shape: tuple[int, ...],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.conv2dInputBackward(
        grad_output._id,
        weight._id,
        list(input_shape),
        list(grad_output_shape),
        list(stride),
        list(padding),
    ))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def conv2d_weight_backward_from_tensors(
    grad_output: "Tensor",
    input: "Tensor",
    weight_shape: tuple[int, ...],
    grad_output_shape: tuple[int, ...],
    input_shape: tuple[int, ...],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.conv2dWeightBackward(
        grad_output._id,
        input._id,
        list(weight_shape),
        list(grad_output_shape),
        list(input_shape),
        list(stride),
        list(padding),
    ))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def conv2d_bias_backward_from_tensors(
    grad_output: "Tensor",
    out_ch: int,
    grad_output_shape: tuple[int, ...],
) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.conv2dBiasBackward(
        grad_output._id,
        int(out_ch),
        list(grad_output_shape),
    ))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def log_softmax_backward_from_tensors(
    grad_output: "Tensor",
    softmax: "Tensor",
    batch_size: int,
    num_classes: int,
) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.logSoftmaxBackward(
        grad_output._id,
        softmax._id,
        int(batch_size),
        int(num_classes),
    ))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def softmax_backward_from_tensors(
    grad_output: "Tensor",
    softmax: "Tensor",
    batch_size: int,
    num_classes: int,
) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.softmaxBackward(
        grad_output._id,
        softmax._id,
        int(batch_size),
        int(num_classes),
    ))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def nll_loss_backward_from_tensors(
    targets: "Tensor",
    batch_size: int,
    num_classes: int,
    scale: float = 1.0,
) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.nllLossBackward(
        targets._id,
        int(batch_size),
        int(num_classes),
        float(scale),
    ))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def slice_backward_from_tensors(
    grad_output: "Tensor",
    input_shape: list[int],
    sliced_shape: list[int],
    dim: int,
    start: int,
    step: int = 1,
) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.sliceBackward(
        grad_output._id,
        list(input_shape),
        list(sliced_shape),
        int(dim),
        int(start),
        int(step),
    ))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def sort_backward_from_tensors(
    grad_output: "Tensor",
    indices: "Tensor",
    input_shape: list[int],
    dim: int,
) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.sortBackward(
        grad_output._id,
        indices._id,
        list(input_shape),
        int(dim),
    ))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def topk_backward_from_tensors(
    grad_output: "Tensor",
    indices: "Tensor",
    input_shape: list[int],
    dim: int,
    k: int,
) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.topkBackward(
        grad_output._id,
        indices._id,
        list(input_shape),
        int(dim),
        int(k),
    ))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def cross_entropy_backward_from_tensors(
    grad_output: "Tensor",
    input: "Tensor",
    targets: "Tensor",
    reduction: str = "mean",
) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.crossEntropyBackward(
        grad_output._id,
        input._id,
        targets._id,
        str(reduction),
    ))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def maxmin_backward_from_tensors(
    input: "Tensor",
    grad_output: "Tensor",
    mode: str,
) -> "Tensor":
    from ._tensor import Tensor
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.maxMinBackward(
        input._id,
        grad_output._id,
        str(mode),
    ))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)
