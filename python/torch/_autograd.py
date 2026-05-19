from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from ._tensor import Tensor


# ── Estado global de graduação ─────────────────────────────────────

_grad_enabled = True


def set_grad_enabled(mode: bool) -> None:
    global _grad_enabled
    _grad_enabled = mode


def is_grad_enabled() -> bool:
    return _grad_enabled


class no_grad:
    """Context manager que desabilita o cálculo de gradientes."""
    def __enter__(self) -> no_grad:
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *args: object) -> None:
        global _grad_enabled
        _grad_enabled = self._prev


class inference_mode:
    """Context manager para modo de inferência (alias para no_grad)."""
    def __init__(self, mode: bool = True) -> None:
        self._mode = mode

    def __enter__(self) -> inference_mode:
        global _grad_enabled
        self._prev = _grad_enabled
        if self._mode:
            _grad_enabled = False
        return self

    def __exit__(self, *args: object) -> None:
        global _grad_enabled
        _grad_enabled = self._prev


# ── Grafo computacional ───────────────────────────────────────────

class _Node:
    """Representa uma operação no grafo computacional."""
    __slots__ = ("tensor", "grad_fn", "parents", "requires_grad")

    def __init__(
        self,
        tensor: Tensor,
        grad_fn: Callable[[Tensor], Sequence[Tensor | None]],
        parents: Sequence[Tensor],
        requires_grad: bool = True,
    ) -> None:
        self.tensor = tensor
        self.grad_fn = grad_fn
        self.parents = parents
        self.requires_grad = requires_grad


# ── Backward engine ───────────────────────────────────────────────

def _backward_from_tensor(
    tensor: Tensor,
    gradient: Tensor | None = None,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> None:
    """Calcula gradientes via backpropagation tape-based."""
    if not tensor._requires_grad:
        raise RuntimeError(
            "element 0 of tensors does not require gradients and "
            "does not have a grad_fn"
        )

    # Se nenhum gradiente foi passado, assume grad = ones_like(tensor)
    if gradient is None:
        if tensor._shape == [] or (len(tensor._shape) == 1 and tensor._shape[0] == 1):
            from ._tensor import tensor_from_data
            gradient = tensor_from_data([1.0], [1], dtype=tensor._dtype)
        else:
            from ._tensor import ones_from_shape
            gradient = ones_from_shape(tensor._shape, dtype=tensor._dtype)

    # Acumular grad no tensor raiz
    if tensor.grad is not None:
        tensor.grad = tensor.grad.add(gradient)
    else:
        tensor.grad = gradient

    # Topological sort (reverso)
    topo: list[_Node] = []
    visited: set[int] = set()

    def _build_topo(node: _Node) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        for parent in node.parents:
            if parent._node is not None:
                _build_topo(parent._node)
        topo.append(node)

    if tensor._node is not None:
        _build_topo(tensor._node)

    # Backward pass
    grads: dict[int, Tensor] = {id(tensor): gradient}

    for node in reversed(topo):
        if id(node.tensor) not in grads:
            continue

        grad_output = grads.pop(id(node.tensor))

        # Executar função de gradiente
        parent_grads = node.grad_fn(grad_output)

        for parent, parent_grad in zip(node.parents, parent_grads):
            if parent_grad is None:
                continue
            if not parent._requires_grad:
                continue

            # Acumular gradiente do pai
            if id(parent) in grads:
                grads[id(parent)] = grads[id(parent)].add(parent_grad)
            else:
                grads[id(parent)] = parent_grad

            # Aplicar hook se registrado
            if parent._backward_hooks:
                for hook in parent._backward_hooks.values():
                    result = hook(parent_grad)
                    if result is not None:
                        if id(parent) in grads:
                            grads[id(parent)] = grads[id(parent)].add(result)
                        else:
                            grads[id(parent)] = result

            # Armazenar gradiente no tensor pai
            if parent.grad is not None:
                parent.grad = parent.grad.add(parent_grad)
            else:
                parent.grad = parent_grad

    # Limpar grafo se não reter
    if not retain_graph:
        _clear_graph(tensor)


def _clear_graph(tensor: Tensor) -> None:
    """Limpa referências do grafo computacional para liberar memória."""
    tensor._node = None
    # Não limpamos parents._node pois podem ser compartilhados


# ── Funções de gradiente (regras de derivada) ─────────────────────

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
    from ._tensor import softmax_from_tensor
    s = softmax_from_tensor(input_tensor, dim)
    # Produto elemento a elemento
    prod = grad_output.mul(s)
    # Soma ao longo da dimensão
    sum_prod = prod.sum()  # simplificado; precisa sum ao longo de dim
    return s.mul(grad_output.sub(sum_prod))


def _grad_log_softmax(grad_output: Tensor, input_tensor: Tensor, dim: int = -1) -> Tensor | None:
    """d/dinput log_softmax(input) = grad_output - softmax(input) * sum(grad_output, dim)"""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import softmax_from_tensor, sum_dim_from_tensor
    s = softmax_from_tensor(input_tensor, dim)
    # Soma do grad_output ao longo da dimensão
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
    from ._tensor import zeros_like_from_tensor, fill_from_tensor
    grad_input = zeros_like_from_tensor(input_tensor)
    # Preencher a fatia selecionada com grad_output
    # Simplificado: precisa de scatter_
    return grad_input


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
    grad_x = grad_output if x._requires_grad else None
    grad_y = grad_output if y._requires_grad else None
    # condition não tem gradiente
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
    from ._tensor import nll_loss_backward_from_tensor, ones_from_tensor

    batch_size = input_tensor.shape[0] if len(input_tensor.shape) > 0 else 1
    num_classes = input_tensor.shape[-1] if len(input_tensor.shape) > 1 else 1

    # Scale depends on reduction: for mean reduction, scale = 1/batch_size
    # For sum reduction, scale = 1.0
    scale = 1.0 / batch_size

    grad_input = nll_loss_backward_from_tensors(target, batch_size, num_classes, scale)

    # Multiply by grad_output (chain rule)
    # If grad_output is scalar, broadcast it
    if grad_output.shape == ():
        scalar_val = grad_output.tolist()[0] if grad_output.tolist() else 1.0
        from ._tensor import _scalar_to_tensor
        grad_input = grad_input.mul(_scalar_to_tensor(scalar_val))
    else:
        grad_input = grad_input.mul(grad_output)

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
    from ._tensor import zeros_like_from_tensor, argmax_from_tensor
    idx = argmax_from_tensor(input_tensor)
    grad_input = zeros_like_from_tensor(input_tensor)
    # Simplificado: precisa scatter_
    return grad_input


def _grad_min(grad_output: Tensor, input_tensor: Tensor) -> Tensor | None:
    """d/dinput min(input) = one_hot(argmin(input)) * grad_output"""
    if not input_tensor._requires_grad:
        return None
    from ._tensor import zeros_like_from_tensor, argmin_from_tensor
    idx = argmin_from_tensor(input_tensor)
    grad_input = zeros_like_from_tensor(input_tensor)
    # Simplificado: precisa scatter_
    return grad_input


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

    # Adicionar dimensões à frente se necessário
    if grad_ndim > target_ndim:
        for _ in range(grad_ndim - target_ndim):
            result = result.sum()

    # Somar ao longo das dimensões expandidas
    for i in range(target_ndim):
        if target_shape[i] == 1 and result._shape[i] != 1:
            result = result.sum()  # simplificado; precisa sum ao longo da dim

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
    # Simplificado
    from ._tensor import zeros_like_from_tensor
    return zeros_like_from_tensor(input_tensor)


def _grad_index_select(grad_output: Tensor, input_tensor: Tensor, dim: int, index: Tensor) -> Tensor | None:
    """d/dinput index_select(input, dim, index) = zeros_like(input); result[dim, index] = grad_output"""
    if not input_tensor._requires_grad:
        return None
    # Simplificado
    from ._tensor import zeros_like_from_tensor
    return zeros_like_from_tensor(input_tensor)
