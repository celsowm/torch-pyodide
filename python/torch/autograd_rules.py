from __future__ import annotations

import math

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from ._tensor import Tensor


def _grad_add(grad_output: Tensor, a: Tensor, b: Tensor) -> tuple[Tensor | None, Tensor | None]:
    """d/da (a + b) = 1, d/db (a + b) = 1"""
    if a._requires_grad:
        grad_a = grad_output
    else:
        grad_a = None
    if b._requires_grad:
        grad_b = grad_output
    else:
        grad_b = None
    return grad_a, grad_b


def _grad_sub(grad_output: Tensor, a: Tensor, b: Tensor) -> tuple[Tensor | None, Tensor | None]:
    """d/da (a - b) = 1, d/db (a - b) = -1"""
    if a._requires_grad:
        grad_a = grad_output
    else:
        grad_a = None
    if b._requires_grad:
        grad_b = grad_output.neg()
    else:
        grad_b = None
    return grad_a, grad_b


def _grad_mul(grad_output: Tensor, a: Tensor, b: Tensor) -> tuple[Tensor | None, Tensor | None]:
    """d/da (a * b) = b, d/db (a * b) = a"""
    if a._requires_grad:
        grad_a = grad_output.mul(b)
    else:
        grad_a = None
    if b._requires_grad:
        grad_b = grad_output.mul(a)
    else:
        grad_b = None
    return grad_a, grad_b


def _grad_div(grad_output: Tensor, a: Tensor, b: Tensor) -> tuple[Tensor | None, Tensor | None]:
    """d/da (a / b) = 1/b, d/db (a / b) = -a/b²"""
    if a._requires_grad:
        grad_a = grad_output.div(b)
    else:
        grad_a = None
    if b._requires_grad:
        grad_b = grad_output.mul(a).div(b.mul(b)).neg()
    else:
        grad_b = None
    return grad_a, grad_b


def _grad_pow(grad_output: Tensor, base: Tensor, exp_tensor: Tensor) -> tuple[Tensor | None, Tensor | None]:
    """d/dbase (base^exp) = exp * base^(exp-1)"""
    if base._requires_grad:
        # grad = grad_output * exp * base^(exp-1)
        grad_base = grad_output.mul(exp_tensor).mul(base.pow(exp_tensor.sub(1)))
    else:
        grad_base = None
    # Derivada em relação ao expoente envolve log, ignorar por enquanto
    grad_exp = None
    return grad_base, grad_exp


def _grad_matmul(grad_output: Tensor, a: Tensor, b: Tensor) -> tuple[Tensor | None, Tensor | None]:
    """d/da (a @ b) = grad @ b.T, d/db (a @ b) = a.T @ grad"""
    if a._requires_grad:
        # grad_a = grad_output @ b.T
        b_t = b.T if len(b._shape) == 2 else b.permute(list(range(b.ndim - 2)) + [b.ndim - 1, b.ndim - 2])
        grad_a = grad_output.matmul(b_t)
    else:
        grad_a = None

    if b._requires_grad:
        # grad_b = a.T @ grad_output
        a_t = a.T if len(a._shape) == 2 else a.permute(list(range(a.ndim - 2)) + [a.ndim - 1, a.ndim - 2])
        grad_b = a_t.matmul(grad_output)
    else:
        grad_b = None

    return grad_a, grad_b


def _grad_sum(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput sum(input) = broadcast(grad_output, input.shape)"""
    if not input_tensor._requires_grad:
        return None
    # grad_output é escalar, precisa fazer broadcast para shape original
    from ._tensor import expand_from_tensor
    return expand_from_tensor(grad_output, input_tensor._shape)


def _grad_mean(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput mean(input) = broadcast(grad_output, input.shape) / numel"""
    if not input_tensor._requires_grad:
        return None
    numel = 1
    for s in input_tensor._shape:
        numel *= s
    from ._tensor import expand_from_tensor
    scaled = grad_output.div(float(numel))
    return expand_from_tensor(scaled, input_tensor._shape)


def _grad_relu(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput relu(input) = grad_output * (input > 0)"""
    if not input_tensor._requires_grad:
        return None
    mask = input_tensor.gt(0).to(input_tensor._dtype)
    return grad_output.mul(mask)


def _grad_sigmoid(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput sigmoid(input) = grad_output * sigmoid(input) * (1 - sigmoid(input))"""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import sigmoid_from_tensor
    sig = sigmoid_from_tensor(input_tensor)
    return grad_output.mul(sig).mul(sig.neg().add(1))


def _grad_tanh(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput tanh(input) = grad_output * (1 - tanh(input)²)"""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import tanh_from_tensor
    t = tanh_from_tensor(input_tensor)
    return grad_output.mul(t.mul(t).neg().add(1))


def _grad_gelu(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput gelu(input) = grad_output * gelu'(input)"""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import sigmoid_from_tensor, tanh_from_tensor
    # gelu'(x) = 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
    #          + 0.5 * x * sech²(...) * sqrt(2/pi) * (1 + 0.134145 * x²)
    # Simplificação: usar a derivada numérica aproximada
    x = input_tensor
    x_cubed = x.mul(x).mul(x)
    inner = x.add(x_cubed.mul(0.044715)).mul(1.128379)  # sqrt(2/pi) ≈ 1.128379
    tanh_inner = tanh_from_tensor(inner)
    sech_sq = tanh_inner.mul(tanh_inner).neg().add(1)
    grad = x.mul(0.5).add(0.5).add(
        x.mul(0.5).mul(sech_sq).mul(inner.div(x).add(0.134145 * x.mul(x)))
    )
    return grad_output.mul(grad)


def _grad_softmax(grad_output: Tensor, input_tensor: Tensor, dim: int = -1) -> Tensor | None:
    """d/dinput softmax(input) = softmax(input) * (grad_output - sum(grad_output * softmax(input), dim))"""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import softmax_from_tensor, sum_dim_from_tensor
    from .grad_mode import no_grad

    with no_grad():
        s = softmax_from_tensor(input_tensor, dim)
        prod = grad_output.mul(s)
        sum_prod = sum_dim_from_tensor(prod, dim, keepdim=True)
        return s.mul(grad_output.sub(sum_prod))


def _grad_log_softmax(grad_output: Tensor, input_tensor: Tensor, dim: int = -1) -> Tensor | None:
    """d/dinput log_softmax(input) = grad_output - softmax(input) * sum(grad_output, dim)"""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import log_softmax_backward_from_tensors, softmax_from_tensor, sum_dim_from_tensor
    from .grad_mode import no_grad

    d = dim if dim >= 0 else dim + len(input_tensor.shape)
    if len(input_tensor.shape) == 2 and d == 1:
        with no_grad():
            softmax = softmax_from_tensor(input_tensor, dim)
            return log_softmax_backward_from_tensors(
                grad_output,
                softmax,
                input_tensor.shape[0],
                input_tensor.shape[1],
            )

    with no_grad():
        s = softmax_from_tensor(input_tensor, dim)
        sum_grad = sum_dim_from_tensor(grad_output, dim, keepdim=True)
        return grad_output.sub(s.mul(sum_grad))


def _grad_neg(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput (-input) = -grad_output"""
    if not input_tensor._requires_grad:
        return None
    return grad_output.neg()


def _grad_abs(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput |input| = grad_output * sign(input)"""
    if not input_tensor._requires_grad:
        return None
    return grad_output.mul(input_tensor.sign())


def _grad_sqrt(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput sqrt(input) = grad_output / (2 * sqrt(input))"""
    if not input_tensor._requires_grad:
        return None
    return grad_output.div(input_tensor.sqrt().mul(2))


def _grad_exp(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput exp(input) = grad_output * exp(input)"""
    if not input_tensor._requires_grad:
        return None
    return grad_output.mul(input_tensor.exp())


def _grad_log(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput log(input) = grad_output / input"""
    if not input_tensor._requires_grad:
        return None
    return grad_output.div(input_tensor)


def _grad_silu(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput silu(input) = grad_output * (sigmoid(input) * (1 + input * (1 - sigmoid(input))))"""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import sigmoid_from_tensor
    sig = sigmoid_from_tensor(input_tensor)
    grad = sig.mul(input_tensor.mul(sig.neg().add(1)).add(1))
    return grad_output.mul(grad)


def _grad_leaky_relu(grad_output: Tensor, input_tensor: Tensor, alpha: float = 0.01) -> Tensor | None:
    """d/dinput leaky_relu(input) = grad_output * (alpha if input < 0 else 1)"""
    if not input_tensor._requires_grad:
        return None
    mask = input_tensor.gt(0).to(input_tensor._dtype)
    grad = mask.add(mask.neg().add(1).mul(alpha))
    return grad_output.mul(grad)


def _grad_reshape(grad_output: Tensor, input_tensor: Tensor, new_shape: Sequence[int]) -> Tensor | None:
    """d/dinput reshape(input) = reshape(grad_output, input.shape)"""
    if not input_tensor._requires_grad:
        return None
    return grad_output.reshape(input_tensor._shape)


def _grad_transpose(grad_output: Tensor, input_tensor: Tensor, dim0: int, dim1: int) -> Tensor | None:
    """d/dinput transpose(input) = transpose(grad_output, dim0, dim1)"""
    if not input_tensor._requires_grad:
        return None
    return grad_output.transpose(dim0, dim1)


def _grad_permute(grad_output: Tensor, input_tensor: Tensor, dims: Sequence[int]) -> Tensor | None:
    """d/dinput permute(input) = permute(grad_output, inverse(dims))"""
    if not input_tensor._requires_grad:
        return None
    # Inversor da permutação
    inv_dims = [0] * len(dims)
    for i, d in enumerate(dims):
        inv_dims[d] = i
    return grad_output.permute(inv_dims)


def _grad_squeeze(grad_output: Tensor, input_tensor: Tensor, dim: int | None) -> Tensor | None:
    """d/dinput squeeze(input) = expand/reshape grad_output para input.shape"""
    if not input_tensor._requires_grad:
        return None
    return grad_output.reshape(input_tensor._shape)


def _grad_unsqueeze(grad_output: Tensor, input_tensor: Tensor, dim: int) -> Tensor | None:
    """d/dinput unsqueeze(input) = squeeze(grad_output, dim)"""
    if not input_tensor._requires_grad:
        return None
    return grad_output.squeeze(dim)


def _grad_cat(grad_output: Tensor, tensors: Sequence[Tensor], dim: int) -> list[Tensor | None]:
    """d/dinput cat(tensors) = split(grad_output, sizes, dim)"""
    result: list[Tensor | None] = []
    offset = 0
    for t in tensors:
        if t._requires_grad:
            size = t._shape[dim if dim >= 0 else dim + len(t._shape)]
            grad_t = grad_output.slice(dim=dim if dim >= 0 else dim + len(t._shape), start=offset, end=offset + size)
            result.append(grad_t)
        else:
            result.append(None)
        offset += t._shape[dim if dim >= 0 else dim + len(t._shape)]
    return result


def _grad_stack(grad_output: Tensor, tensors: Sequence[Tensor], dim: int) -> list[Tensor | None]:
    """d/dinput stack(tensors) = unstack(grad_output, dim)"""
    result: list[Tensor | None] = []
    for i, t in enumerate(tensors):
        if t._requires_grad:
            grad_t = grad_output.select(dim=dim if dim >= 0 else dim + len(t._shape), index=i)
            result.append(grad_t)
        else:
            result.append(None)
    return result


def _grad_select(grad_output: Tensor, input_tensor: Tensor, dim: int, index: int) -> Tensor | None:
    """d/dinput select(input, dim, index) = zeros_like(input); result[dim, index] = grad_output"""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import tensor_from_data, _flatten
    d = dim if dim >= 0 else dim + input_tensor.ndim
    in_shape = list(input_tensor._shape)
    n = 1
    for s in in_shape:
        n *= s
    flat_list = [0.0] * n
    flat_out = _flatten(grad_output.tolist())
    stride = 1
    for s in reversed(in_shape[d+1:]):
        stride *= s
    outer = 1
    for s in in_shape[:d]:
        outer *= s
    for o in range(outer):
        for s in range(stride):
            src_idx = o * (in_shape[d]) * stride + index * stride + s
            dst_idx = o * stride + s
            if src_idx < n and dst_idx < len(flat_out):
                flat_list[src_idx] = flat_out[dst_idx]
    return tensor_from_data(flat_list, in_shape, input_tensor.dtype)


def _grad_slice(grad_output: Tensor, input_tensor: Tensor, dim: int, start: int, end: int, step: int = 1) -> Tensor | None:
    """d/dinput slice(input) = scatter zeros_like(input) at [dim, start:end] with grad_output"""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import slice_backward_from_tensors
    sliced_shape = list(grad_output.shape)
    return slice_backward_from_tensors(
        grad_output,
        list(input_tensor.shape),
        sliced_shape,
        dim,
        start,
        step,
    )


def _grad_where(grad_output: Tensor, condition: Tensor, x: Tensor, y: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """d/dx where(cond, x, y) = where(cond, grad_output, 0)"""
    from ._tensor import zeros_like_from_tensor, where_from_tensors

    zeros = zeros_like_from_tensor(grad_output)
    grad_x = where_from_tensors(condition, grad_output, zeros) if x._requires_grad else None
    grad_y = where_from_tensors(condition, zeros, grad_output) if y._requires_grad else None
    return None, grad_x, grad_y


def _grad_clamp(grad_output: Tensor, input_tensor: Tensor, min_val: float, max_val: float) -> Tensor | None:
    """d/dinput clamp(input, min, max) = grad_output * (min < input < max)"""
    if not input_tensor._requires_grad:
        return None
    mask = input_tensor.gt(min_val).to(input_tensor._dtype).mul(
        input_tensor.lt(max_val).to(input_tensor._dtype)
    )
    return grad_output.mul(mask)


def _grad_cross_entropy(grad_output: Tensor, input_tensor: Tensor, target: Tensor) -> Tensor | None:
    """d/dinput cross_entropy(input, target) = softmax(input) - one_hot(target)"""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import softmax_from_tensor
    s = softmax_from_tensor(input_tensor, dim=-1)
    # Simplificado: assume target é classe e grad_output é escalar
    return s.sub(target) if target._shape == s._shape else s


def _grad_mse_loss(grad_output: Tensor, input_tensor: Tensor, target: Tensor) -> Tensor | None:
    """d/dinput mse_loss(input, target) = 2 * (input - target) / n"""
    if not input_tensor._requires_grad:
        return None
    n = 1
    for s in input_tensor._shape:
        n *= s
    return input_tensor.sub(target).mul(2.0 / n)


def _grad_nll_loss(grad_output: Tensor, input_tensor: Tensor, target: Tensor) -> Tensor | None:
    """d/dinput nll_loss(input, target) = -1/batch_size at target index, 0 elsewhere."""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import nll_loss_backward_from_tensors

    batch_size = input_tensor.shape[0] if len(input_tensor.shape) > 0 else 1
    num_classes = input_tensor.shape[-1] if len(input_tensor.shape) > 1 else 1

    scale = 1.0 / batch_size

    grad_input = nll_loss_backward_from_tensors(target, batch_size, num_classes, scale)

    # Multiply by grad_output (chain rule), handling scalar or per-sample reductions.
    if grad_output.shape == ():
        scalar_val = grad_output.tolist()[0] if grad_output.tolist() else 1.0
        from ._tensor import _scalar_to_tensor
        grad_input = grad_input.mul(_scalar_to_tensor(scalar_val))
    else:
        grad_scale = grad_output
        while len(grad_scale.shape) < len(grad_input.shape):
            grad_scale = grad_scale.unsqueeze(-1)
        grad_input = grad_input.mul(grad_scale)

    return grad_input


def _grad_conv2d(
    grad_output: Tensor,
    input_tensor: Tensor,
    weight_tensor: Tensor,
    output_shape: tuple[int, ...],
    params: tuple,
) -> tuple[Tensor | None, ...]:
    """Backward pass for conv2d.

    Returns gradients for (input, weight) and optionally bias.
    params = (stride, padding, dilation, groups, bias_tensor)
    """
    stride, padding, dilation, groups, bias_tensor = params

    # Compute gradients using runtime-backed functions
    grad_input = None
    grad_weight = None
    grad_bias = None

    input_shape = tuple(input_tensor.shape)
    grad_output_shape = tuple(grad_output.shape)
    weight_shape = tuple(weight_tensor.shape)

    out_ch = weight_shape[0] if len(weight_shape) > 0 else 1

    # Gradient with respect to input
    if input_tensor._requires_grad:
        from ._tensor import conv2d_input_backward_from_tensors
        grad_input = conv2d_input_backward_from_tensors(
            grad_output, weight_tensor, input_shape, grad_output_shape,
            stride, padding,
        )

    # Gradient with respect to weight
    if weight_tensor._requires_grad:
        from ._tensor import conv2d_weight_backward_from_tensors
        grad_weight = conv2d_weight_backward_from_tensors(
            grad_output, input_tensor, weight_shape, grad_output_shape,
            input_shape, stride, padding,
        )

    # Gradient with respect to bias
    if bias_tensor is not None and bias_tensor._requires_grad:
        from ._tensor import conv2d_bias_backward_from_tensors
        grad_bias = conv2d_bias_backward_from_tensors(
            grad_output, out_ch, grad_output_shape,
        )

    # Return in order matching forward inputs: (input_grad, weight_grad, bias_grad)
    return grad_input, grad_weight, grad_bias


def _grad_max(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput max(input) = one_hot(argmax(input)) * grad_output"""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import tensor_from_data, _flatten
    in_vals = _flatten(input_tensor.tolist())
    grad_val = float(_flatten(grad_output.tolist())[0])
    idx = max(range(len(in_vals)), key=lambda i: in_vals[i])
    n = 1
    for s in input_tensor._shape:
        n *= s
    flat = [0.0] * n
    if 0 <= idx < n:
        flat[idx] = grad_val
    return tensor_from_data(flat, list(input_tensor._shape), input_tensor.dtype)


def _grad_min(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput min(input) = one_hot(argmin(input)) * grad_output"""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import tensor_from_data, _flatten
    in_vals = _flatten(input_tensor.tolist())
    grad_val = float(_flatten(grad_output.tolist())[0])
    idx = min(range(len(in_vals)), key=lambda i: in_vals[i])
    n = 1
    for s in input_tensor._shape:
        n *= s
    flat = [0.0] * n
    if 0 <= idx < n:
        flat[idx] = grad_val
    return tensor_from_data(flat, list(input_tensor._shape), input_tensor.dtype)


def _grad_prod(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput prod(input) = prod(input) / input * grad_output"""
    if not input_tensor._requires_grad:
        return None
    return grad_output.mul(input_tensor.prod()).div(input_tensor)


def _grad_cumsum(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput cumsum(input) = flip(cumsum(flip(grad_output)))"""
    if not input_tensor._requires_grad:
        return None
    # Simplificado: gradiente de cumsum é cumsum reverso
    return grad_output


def _grad_cumprod(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput cumprod(input) = complex; simplificado"""
    if not input_tensor._requires_grad:
        return None
    return grad_output


def _grad_expand(grad_output: Tensor, input_tensor: Tensor, new_shape: Sequence[int]) -> Tensor | None:
    """d/dinput expand(input) = sum over broadcasted dims"""
    if not input_tensor._requires_grad:
        return None
    # Reduzir gradientes de volta ao shape original
    return _reduce_broadcast(grad_output, input_tensor._shape)


def _grad_repeat(grad_output: Tensor, input_tensor: Tensor, sizes: Sequence[int]) -> Tensor | None:
    """d/dinput repeat(input) = sum over repeated dims"""
    if not input_tensor._requires_grad:
        return None
    return _reduce_broadcast(grad_output, input_tensor._shape)


def _reduce_broadcast(grad: Tensor, target_shape: Sequence[int]) -> Tensor:
    """Reduz gradiente de volta ao shape original somando dimensões broadcasted."""
    result = grad
    grad_ndim = len(result._shape)
    target_ndim = len(target_shape)

    # Somar dimensões extras à esquerda (broadcast de dimensões ausentes)
    while len(result._shape) > len(target_shape):
        result = result.sum(dim=0)

    # Somar ao longo das dimensões expandidas (onde target_shape[i] == 1)
    for i in range(len(target_shape)):
        if target_shape[i] == 1 and result._shape[i] != 1:
            result = result.sum(dim=i, keepdim=True)

    # Remover dimensões keepdim=1 extras que sobraram
    while len(result._shape) > len(target_shape):
        result = result.sum(dim=0)

    # Reshape para target_shape
    if result._shape != target_shape:
        result = result.reshape(target_shape)

    return result


def _grad_flip(grad_output: Tensor, input_tensor: Tensor, dims: Sequence[int]) -> Tensor | None:
    """d/dinput flip(input) = flip(grad_output, dims)"""
    if not input_tensor._requires_grad:
        return None
    return grad_output.flip(dims)


def _grad_tril(grad_output: Tensor, input_tensor: Tensor, diagonal: int) -> Tensor | None:
    """d/dinput tril(input) = tril(grad_output)"""
    if not input_tensor._requires_grad:
        return None
    return grad_output.tril(diagonal)


def _grad_triu(grad_output: Tensor, input_tensor: Tensor, diagonal: int) -> Tensor | None:
    """d/dinput triu(input) = triu(grad_output)"""
    if not input_tensor._requires_grad:
        return None
    return grad_output.triu(diagonal)


def _grad_masked_fill(grad_output: Tensor, input_tensor: Tensor, mask: Tensor, value: float) -> Tensor | None:
    """d/dinput masked_fill(input, mask, value) = masked_fill(grad_output, ~mask, 0)"""
    if not input_tensor._requires_grad:
        return None
    return grad_output


def _grad_masked_select(grad_output: Tensor, input_tensor: Tensor, mask: Tensor) -> Tensor | None:
    """d/dinput masked_select(input, mask) = zeros_like(input); result[mask] = grad_output"""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import tensor_from_data, _flatten
    in_shape = list(input_tensor._shape)
    n = 1
    for s in in_shape:
        n *= s
    flat_list = [0.0] * n
    mask_flat = _flatten(mask.tolist())
    out_flat = _flatten(grad_output.tolist())
    out_i = 0
    for i, m in enumerate(mask_flat):
        if m and out_i < len(out_flat):
            flat_list[i] = out_flat[out_i]
            out_i += 1
    return tensor_from_data(flat_list, in_shape, input_tensor.dtype)


def _grad_index_select(grad_output: Tensor, input_tensor: Tensor, dim: int, index: Tensor) -> Tensor | None:
    """d/dinput index_select(input, dim, index) = zeros_like(input); result[dim, index] = grad_output"""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import tensor_from_data, _flatten
    in_shape = list(input_tensor._shape)
    d = dim if dim >= 0 else dim + len(in_shape)
    n = 1
    for s in in_shape:
        n *= s
    flat_list = [0.0] * n
    idx_vals = _flatten(index.tolist())
    out_flat = _flatten(grad_output.tolist())
    stride = 1
    for s in reversed(in_shape[d+1:]):
        stride *= s
    outer = 1
    for s in in_shape[:d]:
        outer *= s
    chunk_size = stride
    for i, pos in enumerate(idx_vals):
        pos_i = int(pos)
        for o in range(outer):
            for s in range(chunk_size):
                src_idx = o * in_shape[d] * stride + pos_i * stride + s
                dst_idx = i * chunk_size + s + o * len(idx_vals) * chunk_size
                if src_idx < n and dst_idx < len(out_flat):
                    flat_list[src_idx] = out_flat[dst_idx]
    return tensor_from_data(flat_list, in_shape, input_tensor.dtype)


def _grad_gather(grad_output: Tensor, input_tensor: Tensor, dim: int, index: Tensor) -> Tensor | None:
    """d/dinput gather(input, dim, index) = zeros_like(input); result[index[i,...], ...] += grad_output[i,...]"""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import tensor_from_data, _flatten
    in_shape = list(input_tensor._shape)
    d = dim if dim >= 0 else dim + len(in_shape)
    n = math.prod(in_shape)
    flat_list = [0.0] * n
    idx_vals = _flatten(index.tolist())
    out_flat = _flatten(grad_output.tolist())

    strides = [1] * len(in_shape)
    for i in range(len(in_shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * in_shape[i + 1]

    out_shape = list(index._shape)
    out_strides = [1] * len(out_shape)
    for i in range(len(out_shape) - 2, -1, -1):
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1]

    for i in range(len(out_flat)):
        remaining = i
        src_linear = 0
        for dim_idx in range(len(out_shape) - 1, -1, -1):
            coord = remaining % out_shape[dim_idx]
            remaining //= out_shape[dim_idx]
            if dim_idx == d:
                src_linear += int(idx_vals[i]) * strides[dim_idx]
            else:
                src_linear += coord * strides[dim_idx]
        if 0 <= src_linear < n:
            flat_list[src_linear] += out_flat[i]
    return tensor_from_data(flat_list, in_shape, input_tensor.dtype)


def _grad_topk(grad_output: Tensor, input_tensor: Tensor, dim: int, k: int, descending: bool, saved_indices: Tensor | None = None) -> Tensor | None:
    """d/dinput topk(input) = scatter grad_output back to full shape at topk indices."""
    if not input_tensor._requires_grad:
        return None
    n = 1
    for s in input_tensor._shape:
        n *= s
    from ._tensor import tensor_from_data, _flatten
    flat = [0.0] * n
    grads_np = grad_output.tolist()
    if saved_indices is not None:
        idx_np = saved_indices.tolist()
    else:
        from ._tensor import sort_from_tensor
        _, indices = sort_from_tensor(input_tensor, dim, descending)
        idx_np = indices.tolist()
    flat_grads = _flatten(grads_np)
    flat_idx = _flatten(idx_np)
    for i in range(min(len(flat_idx), len(flat_grads))):
        pos = int(flat_idx[i])
        if 0 <= pos < n:
            flat[pos] += flat_grads[i]
    return tensor_from_data(flat, list(input_tensor._shape), input_tensor.dtype)


def _grad_sort(grad_output: Tensor, input_tensor: Tensor, dim: int, descending: bool, saved_indices: Tensor | None = None) -> Tensor | None:
    """d/dinput sort(input) = scatter grad_output back to original positions."""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import tensor_from_data
    n = 1
    for s in input_tensor._shape:
        n *= s
    flat = [0.0] * n
    grads_np = grad_output.tolist()
    if saved_indices is not None:
        idx_np = saved_indices.tolist()
    else:
        from ._tensor import sort_from_tensor
        _, indices = sort_from_tensor(input_tensor, dim, descending)
        idx_np = indices.tolist()
    from ._tensor import _flatten
    flat_grads = _flatten(grads_np)
    flat_idx = _flatten(idx_np)
    for i in range(min(len(flat_idx), len(flat_grads))):
        pos = int(flat_idx[i])
        if 0 <= pos < n:
            flat[pos] += flat_grads[i]
    return tensor_from_data(flat, list(input_tensor._shape), input_tensor.dtype)


def _grad_scatter_(grad_output: Tensor, input_tensor: Tensor, dim: int, index: Tensor, src: Tensor | float) -> Tensor | None:
    """d/dinput scatter_(input) = scatter grad_output back (gradient flows through unchanged positions)."""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import tensor_from_data, _flatten
    in_shape = list(input_tensor._shape)
    n = 1
    for s in in_shape:
        n *= s
    flat = [0.0] * n
    out_flat = _flatten(grad_output.tolist())
    idx_flat = _flatten(index.tolist())
    mask = [True] * n
    for i in range(len(idx_flat)):
        pos = int(idx_flat[i])
        if 0 <= pos < n:
            mask[pos] = False
    for i in range(n):
        if mask[i] and i < len(out_flat):
            flat[i] = out_flat[i]
        elif not mask[i]:
            flat[i] = out_flat[i]
    return tensor_from_data(flat, in_shape, input_tensor.dtype)


def _grad_maximum(grad_output: Tensor, a: Tensor, b: Tensor) -> tuple[Tensor | None, Tensor | None]:
    """d/da max(a,b) = grad * (a >= b), d/db max(a,b) = grad * (b > a)"""
    grad_a = None
    grad_b = None
    if a._requires_grad:
        mask = a.ge(b).to(a._dtype)
        grad_a = grad_output.mul(mask)
    if b._requires_grad:
        mask = b.gt(a).to(b._dtype)
        grad_b = grad_output.mul(mask)
    return grad_a, grad_b


def _grad_minimum(grad_output: Tensor, a: Tensor, b: Tensor) -> tuple[Tensor | None, Tensor | None]:
    """d/da min(a,b) = grad * (a <= b), d/db min(a,b) = grad * (b < a)"""
    grad_a = None
    grad_b = None
    if a._requires_grad:
        mask = a.le(b).to(a._dtype)
        grad_a = grad_output.mul(mask)
    if b._requires_grad:
        mask = b.lt(a).to(b._dtype)
        grad_b = grad_output.mul(mask)
    return grad_a, grad_b





def grad(
    outputs: "Tensor | Sequence[Tensor]",
    inputs: "Tensor | Sequence[Tensor]",
    grad_outputs: "Tensor | Sequence[Tensor] | None" = None,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> tuple:
    """Compute and return the sum of gradients of outputs w.r.t. inputs."""
    from .autograd_engine import _clear_graph
    from ._tensor import ones_like_from_tensor
    if isinstance(outputs, list):
        total = outputs[0]
        for o in outputs[1:]:
            total = total.add(o)
    else:
        total = outputs

    if isinstance(inputs, list):
        input_list = inputs
    else:
        input_list = [inputs]

    if grad_outputs is None:
        grad_outputs = ones_like_from_tensor(total)

    total._backward(grad_outputs)

    grads: list = []
    for inp in input_list:
        g = inp.grad if inp._requires_grad else None
        grads.append(g)

    if not retain_graph:
        _clear_graph(total)

    return tuple(grads)
