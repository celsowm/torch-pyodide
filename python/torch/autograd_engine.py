from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from ._tensor import Tensor


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

    def _run_backward() -> None:
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

    if create_graph:
        _run_backward()
    else:
        from .grad_mode import no_grad
        with no_grad():
            _run_backward()

    # Limpar grafo se não reter
    if not retain_graph:
        _clear_graph(tensor)


def _clear_graph(tensor: Tensor) -> None:
    """Limpa referências do grafo computacional para liberar memória."""
    tensor._node = None
    # Não limpamos parents._node pois podem ser compartilhados



