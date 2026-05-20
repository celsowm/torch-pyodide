from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence

from ._runtime import _get_runtime, _run_js_awaitable

if TYPE_CHECKING:
    from .autograd import _Node


def _js_meta_to_tuple(meta: object) -> tuple[int, list[int], str]:
    if isinstance(meta, dict):
        tensor_id = int(meta["id"])
        shape_raw = meta["shape"]
        shape = list(shape_raw.to_py() if hasattr(shape_raw, "to_py") else shape_raw)
        dtype = str(meta["dtype"])
        return tensor_id, shape, dtype
    tensor_id = int(getattr(meta, "id"))
    shape_raw = getattr(meta, "shape")
    shape = list(shape_raw.to_py() if hasattr(shape_raw, "to_py") else shape_raw)
    dtype = str(getattr(meta, "dtype"))
    return tensor_id, shape, dtype


@dataclass(slots=True)
class Tensor:
    _id: int
    _shape: list[int]
    _dtype: str
    _requires_grad: bool = False
    _node: "_Node | None" = field(default=None, repr=False)
    _backward_hooks: dict[int, Callable] = field(default_factory=dict, repr=False)
    grad: "Tensor | None" = field(default=None, repr=False)
    _retains_grad: bool = field(default=False, repr=False)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._shape)

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def numel(self) -> int:
        n = 1
        for s in self._shape:
            n *= s
        return n

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = value

    def requires_grad_(self, requires_grad: bool = True) -> "Tensor":
        """Sets this tensor's requires_grad attribute in-place."""
        self._requires_grad = requires_grad
        return self

    @property
    def is_leaf(self) -> bool:
        """Um tensor é leaf se não foi criado por uma operação (não tem _node)."""
        return self._node is None

    def register_hook(self, hook: Callable[["Tensor"], "Tensor | None"]) -> int:
        """Registra um hook para ser chamado no gradiente durante backward().
        
        O hook recebe o gradiente e pode retornar um novo gradiente modificado
        ou None para usar o gradiente original.
        """
        hook_id = id(hook)
        self._backward_hooks[hook_id] = hook
        return hook_id

    def remove_hook(self, hook_id: int) -> None:
        """Remove um hook registrado."""
        self._backward_hooks.pop(hook_id, None)

    def retain_grad(self) -> None:
        """Faz com que gradientes de tensores não-leaf sejam retidos."""
        self._retains_grad = True

    def backward(
        self,
        gradient: "Tensor | None" = None,
        retain_graph: bool = False,
        create_graph: bool = False,
        inputs: Sequence["Tensor"] | None = None,
    ) -> None:
        """Calcula gradientes via backpropagation.
        
        Args:
            gradient: Gradiente inicial (default: ones_like para tensores escalares).
            retain_graph: Se True, mantém o grafo para chamadas posteriores.
            create_graph: Se True, cria grafo para gradientes de alta ordem.
            inputs: Tensores específicos para calcular gradientes (opcional).
        """
        from .autograd import _backward_from_tensor
        _backward_from_tensor(self, gradient, retain_graph, create_graph)

    def to(self, dtype: str, device: object = None) -> "Tensor":
        if dtype is not None and dtype != self._dtype:
            from ._tensor import zeros_like_from_tensor
            empty = zeros_like_from_tensor(self, dtype)
            return empty.add(self)
        return self

    def clone(self) -> "Tensor":
        return self.add(0.0)

    def detach(self) -> "Tensor":
        """Retorna um novo tensor desconectado do grafo computacional."""
        t = Tensor(self._id, list(self._shape), self._dtype)
        # Copia o ID mas não o node - fica desconectado do grafo
        return t

    def detach_(self) -> "Tensor":
        """Desconecta este tensor do grafo computacional in-place."""
        self._node = None
        self.grad = None
        return self

    def contiguous(self) -> "Tensor":
        return self

    cpu = detach
    cuda = detach

    def half(self) -> "Tensor":
        """Convert to float16. Note: WebGPU runtime uses float32 internally, so this is a dtype hint only."""
        return Tensor(self._id, self._shape, "float16", _requires_grad=self._requires_grad)

    def bfloat16(self) -> "Tensor":
        """Convert to bfloat16. Note: WebGPU runtime uses float32 internally, so this is a dtype hint only."""
        return Tensor(self._id, self._shape, "bfloat16", _requires_grad=self._requires_grad)

    def float(self) -> "Tensor":
        return self if self._dtype == "float32" else self.to("float32")

    def double(self) -> "Tensor":
        return self.to("float64") if self._dtype != "float64" else self

    def long(self) -> "Tensor":
        return self.to("int64")

    def int(self) -> "Tensor":
        return self.to("int32")

    def byte(self) -> "Tensor":
        return self.to("uint8")

    def bool(self) -> "Tensor":
        return self.to("bool")

    def __bool__(self) -> bool:
        return self.item() != 0

    def __len__(self) -> int:
        return self._shape[0] if self._shape else 0

    def zero_(self) -> "Tensor":
        return self

    def fill_(self, value: float) -> "Tensor":
        return self

    def copy_(self, src: "Tensor") -> "Tensor":
        return src

    def narrow(self, dim: int, start: int, length: int) -> "Tensor":
        return self.slice(dim=dim, start=start, end=start + length)

    def repeat(self, *sizes: int) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.repeat(self._id, [int(s) for s in sizes]))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def repeat_interleave(self, repeats: int, dim: int | None = None) -> "Tensor":
        from .__init__ import arange
        if dim is None:
            flat = self.flatten()
            shape_0 = flat._shape[0]
            indices = arange(shape_0, dtype="int64").unsqueeze(1).expand(shape_0, repeats).flatten()
            return flat.index_select(0, indices)
        d = dim if dim >= 0 else dim + self._shape.__len__()
        shape_d = self._shape[d]
        indices = arange(shape_d, dtype="int64").unsqueeze(1).expand(shape_d, repeats).flatten()
        return self.index_select(d, indices)

    tile = repeat

    def topk(self, k: int, dim: int = -1, largest: bool = True) -> tuple["Tensor", "Tensor"]:
        from ._tensor import topk_from_tensor
        return topk_from_tensor(self, k, dim, largest)

    def sort(self, dim: int = -1, descending: bool = False) -> tuple["Tensor", "Tensor"]:
        from ._tensor import sort_from_tensor
        return sort_from_tensor(self, dim, descending)

    def gather(self, dim: int, index: "Tensor") -> "Tensor":
        from ._tensor import gather_from_tensor
        return gather_from_tensor(self, dim, index)

    def scatter_(self, dim: int, index: "Tensor", src: "Tensor | float") -> "Tensor":
        from ._tensor import scatter_from_tensor
        return scatter_from_tensor(self, dim, index, src)

    def argsort(self, dim: int = -1, descending: bool = False) -> "Tensor":
        _, indices = sort_from_tensor(self, dim, descending)
        return indices

    def __add__(self, other: "Tensor | float") -> "Tensor":
        return self.add(other)

    def __radd__(self, other: "Tensor | float") -> "Tensor":
        return self.add(other) if isinstance(other, Tensor) else Tensor(0, [1], self._dtype).add(self)

    def __mul__(self, other: "Tensor | float") -> "Tensor":
        return self.mul(other)

    def __rmul__(self, other: "Tensor | float") -> "Tensor":
        return self.mul(other)

    def __sub__(self, other: "Tensor | float") -> "Tensor":
        return self.sub(other)

    def __rsub__(self, other: "Tensor | float") -> "Tensor":
        from ._tensor import _scalar_to_tensor
        return self.neg().add(other) if isinstance(other, Tensor) else _scalar_to_tensor(float(other), self._dtype).sub(self)

    def __truediv__(self, other: "Tensor | float") -> "Tensor":
        return self.div(other)

    def __neg__(self) -> "Tensor":
        return self.neg()

    def __lt__(self, other: "Tensor | float") -> "Tensor":
        from ._tensor import eq_from_tensors, ne_from_tensors, lt_from_tensors, le_from_tensors, gt_from_tensors, ge_from_tensors
        if isinstance(other, float):
            other = _scalar_to_tensor(other, self._dtype)
        return lt_from_tensors(self, other)

    def __le__(self, other: "Tensor | float") -> "Tensor":
        from ._tensor import le_from_tensors
        if isinstance(other, float):
            other = _scalar_to_tensor(other, self._dtype)
        return le_from_tensors(self, other)

    def __gt__(self, other: "Tensor | float") -> "Tensor":
        from ._tensor import gt_from_tensors
        if isinstance(other, float):
            other = _scalar_to_tensor(other, self._dtype)
        return gt_from_tensors(self, other)

    def __ge__(self, other: "Tensor | float") -> "Tensor":
        from ._tensor import ge_from_tensors
        if isinstance(other, float):
            other = _scalar_to_tensor(other, self._dtype)
        return ge_from_tensors(self, other)

    def __eq__(self, other: "Tensor | float") -> "Tensor":
        from ._tensor import eq_from_tensors
        if isinstance(other, float):
            other = _scalar_to_tensor(other, self._dtype)
        return eq_from_tensors(self, other)

    def __ne__(self, other: "Tensor | float") -> "Tensor":
        from ._tensor import ne_from_tensors
        if isinstance(other, float):
            other = _scalar_to_tensor(other, self._dtype)
        return ne_from_tensors(self, other)

    def __pow__(self, other: "Tensor | float") -> "Tensor":
        return pow_from_tensors(self, other) if isinstance(other, Tensor) else pow_from_tensors(self, _scalar_to_tensor(float(other), self._dtype))

    def __and__(self, other: "Tensor") -> "Tensor":
        return self.mul(other)

    def __or__(self, other: "Tensor") -> "Tensor":
        return self.add(other).clamp(0.0, 1.0)

    def __xor__(self, other: "Tensor") -> "Tensor":
        return (self.__or__(other)).__sub__(self.__and__(other))

    def __invert__(self) -> "Tensor":
        return _scalar_to_tensor(1.0, self._dtype).sub(self)

    def matmul(self, other: "Tensor") -> "Tensor":
        from .autograd import _Node, is_grad_enabled, _grad_matmul

        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.matmul(self._id, other._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)

        requires_grad = is_grad_enabled() and (self._requires_grad or other._requires_grad)
        if requires_grad:
            result_tensor = Tensor(tensor_id, shape, dtype, _requires_grad=True)
            node = _Node(
                tensor=result_tensor,
                grad_fn=lambda grad_out: _grad_matmul(grad_out, self, other),
                parents=[self, other],
            )
            result_tensor._node = node
            return result_tensor

        return Tensor(tensor_id, shape, dtype)

    def mm(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    def bmm(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    def mv(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    def dot(self, other: "Tensor") -> "Tensor":
        return (self * other).sum()

    def outer(self, other: "Tensor") -> "Tensor":
        return self.reshape(-1, 1) * other.reshape(1, -1)

    def norm(self, p: float | str = "fro") -> "Tensor":
        if p == "fro" or p == 2:
            return (self * self).sum().sqrt()
        elif p == 1:
            return self.abs().sum()
        elif p == float("inf") or p == "inf":
            return self.abs().max()
        else:
            return (self.abs() ** p).sum() ** (1.0 / p)

    def cholesky(self) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.cholesky(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def add(self, other: "Tensor | float") -> "Tensor":
        from ._tensor import _get_runtime, _run_js_awaitable, _js_meta_to_tuple, Tensor, _scalar_to_tensor
        from .autograd import _Node, is_grad_enabled, _grad_add

        runtime = _get_runtime()
        b_tensor = other if isinstance(other, Tensor) else _scalar_to_tensor(float(other), self._dtype)
        b = b_tensor._id
        meta = _run_js_awaitable(runtime.add(self._id, b))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)

        # Registrar node no grafo se graduação estiver habilitada
        requires_grad = is_grad_enabled() and (self._requires_grad or (isinstance(other, Tensor) and other._requires_grad))
        if requires_grad:
            result_tensor = Tensor(tensor_id, shape, dtype, _requires_grad=True)
            node = _Node(
                tensor=result_tensor,
                grad_fn=lambda grad_out: _grad_add(grad_out, self, b_tensor),
                parents=[self, b_tensor],
            )
            result_tensor._node = node
            return result_tensor

        return Tensor(tensor_id, shape, dtype)

    def sub(self, other: "Tensor | float") -> "Tensor":
        from ._tensor import _get_runtime, _run_js_awaitable, _js_meta_to_tuple, Tensor, _scalar_to_tensor
        from .autograd import _Node, is_grad_enabled, _grad_sub

        runtime = _get_runtime()
        b_tensor = other if isinstance(other, Tensor) else _scalar_to_tensor(float(other), self._dtype)
        b = b_tensor._id
        meta = _run_js_awaitable(runtime.sub(self._id, b))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)

        requires_grad = is_grad_enabled() and (self._requires_grad or (isinstance(other, Tensor) and other._requires_grad))
        if requires_grad:
            result_tensor = Tensor(tensor_id, shape, dtype, _requires_grad=True)
            node = _Node(
                tensor=result_tensor,
                grad_fn=lambda grad_out: _grad_sub(grad_out, self, b_tensor),
                parents=[self, b_tensor],
            )
            result_tensor._node = node
            return result_tensor

        return Tensor(tensor_id, shape, dtype)

    def mul(self, other: "Tensor | float") -> "Tensor":
        from ._tensor import _get_runtime, _run_js_awaitable, _js_meta_to_tuple, Tensor, _scalar_to_tensor
        from .autograd import _Node, is_grad_enabled, _grad_mul

        runtime = _get_runtime()
        b_tensor = other if isinstance(other, Tensor) else _scalar_to_tensor(float(other), self._dtype)
        b = b_tensor._id
        meta = _run_js_awaitable(runtime.mul(self._id, b))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)

        requires_grad = is_grad_enabled() and (self._requires_grad or (isinstance(other, Tensor) and other._requires_grad))
        if requires_grad:
            result_tensor = Tensor(tensor_id, shape, dtype, _requires_grad=True)
            node = _Node(
                tensor=result_tensor,
                grad_fn=lambda grad_out: _grad_mul(grad_out, self, b_tensor),
                parents=[self, b_tensor],
            )
            result_tensor._node = node
            return result_tensor

        return Tensor(tensor_id, shape, dtype)

    def div(self, other: "Tensor | float") -> "Tensor":
        from ._tensor import _get_runtime, _run_js_awaitable, _js_meta_to_tuple, Tensor, _scalar_to_tensor
        from .autograd import _Node, is_grad_enabled, _grad_div

        runtime = _get_runtime()
        b_tensor = other if isinstance(other, Tensor) else _scalar_to_tensor(float(other), self._dtype)
        b = b_tensor._id
        meta = _run_js_awaitable(runtime.div(self._id, b))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)

        requires_grad = is_grad_enabled() and (self._requires_grad or (isinstance(other, Tensor) and other._requires_grad))
        if requires_grad:
            result_tensor = Tensor(tensor_id, shape, dtype, _requires_grad=True)
            node = _Node(
                tensor=result_tensor,
                grad_fn=lambda grad_out: _grad_div(grad_out, self, b_tensor),
                parents=[self, b_tensor],
            )
            result_tensor._node = node
            return result_tensor

        return Tensor(tensor_id, shape, dtype)

    def lu(self) -> tuple["Tensor", "Tensor"]:
        runtime = _get_runtime()
        result = _run_js_awaitable(runtime.lu(self._id))
        a_meta = result[0]
        p_meta = result[1]
        a_id, a_shape, a_dtype = _js_meta_to_tuple(a_meta)
        p_id, p_shape, p_dtype = _js_meta_to_tuple(p_meta)
        return Tensor(a_id, a_shape, a_dtype), Tensor(p_id, p_shape, p_dtype)

    def triangular_solve(self, b: "Tensor", upper: bool = False) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.triangularSolve(self._id, b._id, upper))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def item(self) -> float:
        return _run_js_awaitable(_get_runtime().toList(self._id))[0]

    def det(self) -> "Tensor":
        from ._tensor import tensor_from_data
        n = self._shape[-1]
        a_lu, pivot = self.lu()
        pivot_data = _run_js_awaitable(_get_runtime().toList(pivot._id))
        pivot_data = [int(x) for x in pivot_data]
        visited = [False] * n
        sign = 1
        for i in range(n):
            if not visited[i]:
                j = i
                while not visited[j]:
                    visited[j] = True
                    j = pivot_data[j]
                if j != i:
                    sign *= -1
        a_data = _run_js_awaitable(_get_runtime().toList(a_lu._id))
        u_diag_prod = 1.0
        for i in range(n):
            u_diag_prod *= a_data[i * n + i]
        return tensor_from_data(sign * u_diag_prod, self._dtype)

    def inv(self) -> "Tensor":
        from .__init__ import zeros, tril, triu, cat, ones
        from ._tensor import tensor_from_data
        n = self._shape[-1]
        a_lu, pivot = self.lu()
        # Extract L: tril with diagonal=-1 + identity
        l_part = tril(a_lu, diagonal=-1)
        # Create identity for diagonal of L
        eye_data = [0.0] * (n * n)
        for i in range(n):
            eye_data[i * n + i] = 1.0
        l_full = tensor_from_data(eye_data, self._dtype).reshape([n, n]) + l_part
        # Extract U: triu with diagonal=0
        u_full = triu(a_lu, diagonal=0)
        # Solve for each column of identity
        inv_cols = []
        for j in range(n):
            col = tensor_from_data([1.0 if i == j else 0.0 for i in range(n)], self._dtype).reshape([n, 1])
            y = l_full.triangular_solve(col, upper=False)
            x = u_full.triangular_solve(y, upper=True)
            inv_cols.append(x)
        return cat(inv_cols, dim=1)

    def diag(self) -> "Tensor":
        from ._tensor import tensor_from_data, _flatten
        flat_list_raw = self.tolist()
        if isinstance(flat_list_raw, list):
            flat_list: list[float] = _flatten(flat_list_raw)
        else:
            flat_list = [float(flat_list_raw)]
        if len(self._shape) == 1:
            n = self._shape[0]
            data: list[float] = [0.0] * (n * n)
            for i in range(n):
                data[i * n + i] = flat_list[i]
            return tensor_from_data(data, [n, n], self._dtype)
        n = self._shape[-1]
        nrows = self._shape[0]
        result_data = [flat_list[i * n + i] for i in range(min(nrows, n))]
        return tensor_from_data(result_data, self._dtype)

    def sum(self) -> "Tensor":
        return sum_from_tensor(self)

    def mean(self) -> "Tensor":
        return mean_from_tensor(self)

    def max(self) -> "Tensor":
        return max_from_tensor(self)

    def min(self) -> "Tensor":
        return min_from_tensor(self)

    def prod(self) -> "Tensor":
        return prod_from_tensor(self)

    def pow(self, other: "Tensor | float") -> "Tensor":
        return pow_from_tensors(self, other) if isinstance(other, Tensor) else pow_from_tensors(self, _scalar_to_tensor(float(other), self._dtype))

    def heaviside(self, values: "Tensor") -> "Tensor":
        return heaviside_from_tensors(self, values)

    def maximum(self, other: "Tensor") -> "Tensor":
        return maximum_from_tensors(self, other)

    def minimum(self, other: "Tensor") -> "Tensor":
        return minimum_from_tensors(self, other)

    def any(self) -> "Tensor":
        return any_from_tensor(self)

    def all(self) -> "Tensor":
        return all_from_tensor(self)

    def cumsum(self) -> "Tensor":
        return cumsum_from_tensor(self)

    def cumprod(self) -> "Tensor":
        return cumprod_from_tensor(self)

    def tril(self, diagonal: int = 0) -> "Tensor":
        return tril_from_tensor(self, diagonal)

    def triu(self, diagonal: int = 0) -> "Tensor":
        return triu_from_tensor(self, diagonal)

    def flip(self, dims: Sequence[int]) -> "Tensor":
        return flip_from_tensor(self, dims)

    def where(self, condition: "Tensor", other: "Tensor") -> "Tensor":
        return where_from_tensors(condition, self, other)

    def cat(self, other: "Tensor", dim: int = 0) -> "Tensor":
        return cat_from_tensors([self, other], dim)

    def stack(self, other: "Tensor", dim: int = 0) -> "Tensor":
        return stack_from_tensors([self, other], dim)

    def index_select(self, dim: int, index: "Tensor") -> "Tensor":
        return index_select_from_tensor(self, dim, index)

    def masked_select(self, mask: "Tensor") -> "Tensor":
        return masked_select_from_tensor(self, mask)

    def masked_fill(self, mask: "Tensor", value: float) -> "Tensor":
        return masked_fill_from_tensor(self, mask, value)

    def eq(self, other: "Tensor") -> "Tensor":
        return eq_from_tensors(self, other)

    def ne(self, other: "Tensor") -> "Tensor":
        return ne_from_tensors(self, other)

    def gt(self, other: "Tensor") -> "Tensor":
        return gt_from_tensors(self, other)

    def lt(self, other: "Tensor") -> "Tensor":
        return lt_from_tensors(self, other)

    def ge(self, other: "Tensor") -> "Tensor":
        return ge_from_tensors(self, other)

    def le(self, other: "Tensor") -> "Tensor":
        return le_from_tensors(self, other)

    def isinf(self) -> "Tensor":
        return isinf_from_tensor(self)

    def isfinite(self) -> "Tensor":
        return isfinite_from_tensor(self)

    def isposinf(self) -> "Tensor":
        return isposinf_from_tensor(self)

    def isneginf(self) -> "Tensor":
        return isneginf_from_tensor(self)

    def logical_not(self) -> "Tensor":
        return logical_not_from_tensor(self)

    def erf(self) -> "Tensor":
        return erf_from_tensor(self)

    def erfc(self) -> "Tensor":
        return erfc_from_tensor(self)

    def lgamma(self) -> "Tensor":
        return lgamma_from_tensor(self)

    def digamma(self) -> "Tensor":
        return digamma_from_tensor(self)

    def i0(self) -> "Tensor":
        return i0_from_tensor(self)

    def deg2rad(self) -> "Tensor":
        return deg2rad_from_tensor(self)

    def rad2deg(self) -> "Tensor":
        return rad2deg_from_tensor(self)

    def empty_like(self) -> "Tensor":
        return empty_like_from_tensor(self)

    def zeros_like(self) -> "Tensor":
        return zeros_like_from_tensor(self)

    def ones_like(self) -> "Tensor":
        return ones_like_from_tensor(self)

    def relu(self) -> "Tensor":
        from .autograd import _Node, is_grad_enabled, _grad_relu

        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.relu(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)

        if is_grad_enabled() and self._requires_grad:
            result = Tensor(tensor_id, shape, dtype, _requires_grad=True)
            result._node = _Node(result, lambda g: (_grad_relu(g, self),), [self])
            return result
        return Tensor(tensor_id, shape, dtype)

    def abs(self) -> "Tensor":
        from .autograd import _Node, is_grad_enabled, _grad_abs

        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.abs(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)

        if is_grad_enabled() and self._requires_grad:
            result = Tensor(tensor_id, shape, dtype, _requires_grad=True)
            result._node = _Node(result, lambda g: (_grad_abs(g, self),), [self])
            return result
        return Tensor(tensor_id, shape, dtype)

    def sqrt(self) -> "Tensor":
        from .autograd import _Node, is_grad_enabled, _grad_sqrt

        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.sqrt(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)

        if is_grad_enabled() and self._requires_grad:
            result = Tensor(tensor_id, shape, dtype, _requires_grad=True)
            result._node = _Node(result, lambda g: (_grad_sqrt(g, self),), [self])
            return result
        return Tensor(tensor_id, shape, dtype)

    def exp(self) -> "Tensor":
        from .autograd import _Node, is_grad_enabled, _grad_exp

        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.exp(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)

        if is_grad_enabled() and self._requires_grad:
            result = Tensor(tensor_id, shape, dtype, _requires_grad=True)
            result._node = _Node(result, lambda g: (_grad_exp(g, self),), [self])
            return result
        return Tensor(tensor_id, shape, dtype)

    def log(self) -> "Tensor":
        from .autograd import _Node, is_grad_enabled, _grad_log

        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.log(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)

        if is_grad_enabled() and self._requires_grad:
            result = Tensor(tensor_id, shape, dtype, _requires_grad=True)
            result._node = _Node(result, lambda g: (_grad_log(g, self),), [self])
            return result
        return Tensor(tensor_id, shape, dtype)

    def neg(self) -> "Tensor":
        from .autograd import _Node, is_grad_enabled, _grad_neg

        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.neg(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)

        if is_grad_enabled() and self._requires_grad:
            result = Tensor(tensor_id, shape, dtype, _requires_grad=True)
            result._node = _Node(result, lambda g: (_grad_neg(g, self),), [self])
            return result
        return Tensor(tensor_id, shape, dtype)

    def clamp(self, min: float, max: float) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.clamp(self._id, float(min), float(max)))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def argmax(self) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.argmax(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def argmin(self) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.argmin(self._id))
        tensor_id, shape, dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, shape, dtype)

    def reshape(self, shape: int | Sequence[int]) -> "Tensor":
        runtime = _get_runtime()
        normalized = _normalize_shape(shape)
        meta = _run_js_awaitable(runtime.reshape(self._id, normalized))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def view(self, *shape: int) -> "Tensor":
        normalized = _normalize_shape_from_args(shape)
        return self.reshape(normalized)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.flatten(self._id, int(start_dim), int(end_dim)))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def squeeze(self, dim: int | None = None) -> "Tensor":
        runtime = _get_runtime()
        if dim is None:
            meta = _run_js_awaitable(runtime.squeeze(self._id))
        else:
            meta = _run_js_awaitable(runtime.squeeze(self._id, int(dim)))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def unsqueeze(self, dim: int) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.unsqueeze(self._id, int(dim)))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.transpose(self._id, int(dim0), int(dim1)))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def permute(self, dims: Sequence[int]) -> "Tensor":
        runtime = _get_runtime()
        normalized = [int(v) for v in dims]
        meta = _run_js_awaitable(runtime.permute(self._id, normalized))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def expand(self, *shape: int) -> "Tensor":
        runtime = _get_runtime()
        normalized = _normalize_shape_from_args(shape)
        meta = _run_js_awaitable(runtime.expand(self._id, normalized))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def select(self, dim: int, index: int) -> "Tensor":
        from .autograd import _Node, is_grad_enabled, _grad_select

        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.select(self._id, int(dim), int(index)))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

        if is_grad_enabled() and self._requires_grad:
            result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
            result._node = _Node(result, lambda g: (_grad_select(g, self, dim, index),), [self])
            return result
        return Tensor(tensor_id, out_shape, out_dtype)

    def slice(self, dim: int, start: int | None = None, end: int | None = None, step: int = 1) -> "Tensor":
        from .autograd import _Node, is_grad_enabled, _grad_slice

        runtime = _get_runtime()
        resolved_start = start if start is not None else 0
        resolved_end = end
        if start is None and end is None:
            meta = _run_js_awaitable(runtime.slice(self._id, int(dim), None, None, int(step)))
        elif end is None:
            meta = _run_js_awaitable(runtime.slice(self._id, int(dim), int(start), None, int(step)))
        else:
            meta = _run_js_awaitable(runtime.slice(self._id, int(dim), int(start) if start is not None else None, int(end), int(step)))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

        if is_grad_enabled() and self._requires_grad:
            result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
            actual_start = resolved_start if start is not None else 0
            actual_step = step
            result._node = _Node(result, lambda g: (_grad_slice(g, self, dim, actual_start, resolved_end, actual_step),), [self])
            return result
        return Tensor(tensor_id, out_shape, out_dtype)

    @property
    def T(self) -> "Tensor":
        runtime = _get_runtime()
        meta = _run_js_awaitable(runtime.transpose2d(self._id))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)

    def tolist(self) -> object:
        runtime = _get_runtime()
        result = _run_js_awaitable(runtime.toList(self._id))
        flat = list(result.to_py() if hasattr(result, "to_py") else result)
        return _reshape_flat_values(flat, self._shape, self._dtype)

    def split(self, split_size: int | list[int], dim: int = 0) -> list["Tensor"]:
        shape = self._shape
        d = dim if dim >= 0 else dim + len(shape)
        size_dim = shape[d]
        if isinstance(split_size, int):
            sections = []
            i = 0
            while i < size_dim:
                end = min(i + split_size, size_dim)
                sections.append(end - i)
                i = end
        else:
            sections = [int(s) for s in split_size]
        result: list[Tensor] = []
        offset = 0
        for sec in sections:
            result.append(self.slice(dim=d, start=offset, end=offset + sec))
            offset += sec
        return result

    def chunk(self, chunks: int, dim: int = 0) -> list["Tensor"]:
        shape = self._shape
        d = dim if dim >= 0 else dim + len(shape)
        size_dim = shape[d]
        split_size = (size_dim + chunks - 1) // chunks
        return self.split(split_size, dim=dim)

    def destroy(self) -> None:
        runtime = _get_runtime()
        _run_js_awaitable(runtime.destroy(self._id))

    def sigmoid(self) -> "Tensor":
        return sigmoid_from_tensor(self)

    def tanh(self) -> "Tensor":
        return tanh_from_tensor(self)

    def sin(self) -> "Tensor":
        return sin_from_tensor(self)

    def cos(self) -> "Tensor":
        return cos_from_tensor(self)

    def gelu(self) -> "Tensor":
        return gelu_from_tensor(self)

    def silu(self) -> "Tensor":
        return silu_from_tensor(self)

    def softmax(self, dim: int = -1) -> "Tensor":
        return softmax_from_tensor(self, dim)

    def log_softmax(self, dim: int = -1) -> "Tensor":
        return log_softmax_from_tensor(self, dim)

    def leaky_relu(self, alpha: float = 0.01) -> "Tensor":
        return leaky_relu_from_tensor(self, alpha)

    def floor(self) -> "Tensor":
        return floor_from_tensor(self)

    def ceil(self) -> "Tensor":
        return ceil_from_tensor(self)

    def round(self) -> "Tensor":
        return round_from_tensor(self)

    def reciprocal(self) -> "Tensor":
        return reciprocal_from_tensor(self)

    def square(self) -> "Tensor":
        return square_from_tensor(self)

    # ── Fase 0: unary ops ──────────────────────────────────────────
    def tan(self) -> "Tensor":
        return tan_from_tensor(self)

    def asin(self) -> "Tensor":
        return asin_from_tensor(self)

    def acos(self) -> "Tensor":
        return acos_from_tensor(self)

    def atan(self) -> "Tensor":
        return atan_from_tensor(self)

    def sinh(self) -> "Tensor":
        return sinh_from_tensor(self)

    def cosh(self) -> "Tensor":
        return cosh_from_tensor(self)

    def asinh(self) -> "Tensor":
        return asinh_from_tensor(self)

    def acosh(self) -> "Tensor":
        return acosh_from_tensor(self)

    def atanh(self) -> "Tensor":
        return atanh_from_tensor(self)

    def exp2(self) -> "Tensor":
        return exp2_from_tensor(self)

    def log2(self) -> "Tensor":
        return log2_from_tensor(self)

    def log10(self) -> "Tensor":
        return log10_from_tensor(self)

    def log1p(self) -> "Tensor":
        return log1p_from_tensor(self)

    def expm1(self) -> "Tensor":
        return expm1_from_tensor(self)

    def trunc(self) -> "Tensor":
        return trunc_from_tensor(self)

    def frac(self) -> "Tensor":
        return frac_from_tensor(self)

    def softplus(self) -> "Tensor":
        return softplus_from_tensor(self)

    def mish(self) -> "Tensor":
        return mish_from_tensor(self)

    def hardsigmoid(self) -> "Tensor":
        return hardsigmoid_from_tensor(self)

    def hardswish(self) -> "Tensor":
        return hardswish_from_tensor(self)

    def softsign(self) -> "Tensor":
        return softsign_from_tensor(self)

    def tanhshrink(self) -> "Tensor":
        return tanhshrink_from_tensor(self)

    def rsqrt(self) -> "Tensor":
        return rsqrt_from_tensor(self)

    def sign(self) -> "Tensor":
        return sign_from_tensor(self)

    def sgn(self) -> "Tensor":
        return sgn_from_tensor(self)

    def isnan(self) -> "Tensor":
        return isnan_from_tensor(self)

    def isinf(self) -> "Tensor":
        return isinf_from_tensor(self)

    def isfinite(self) -> "Tensor":
        return isfinite_from_tensor(self)

    def isposinf(self) -> "Tensor":
        return isposinf_from_tensor(self)

    def isneginf(self) -> "Tensor":
        return isneginf_from_tensor(self)

    def logical_not(self) -> "Tensor":
        return logical_not_from_tensor(self)

    def erf(self) -> "Tensor":
        return erf_from_tensor(self)

    def erfc(self) -> "Tensor":
        return erfc_from_tensor(self)

    def lgamma(self) -> "Tensor":
        return lgamma_from_tensor(self)

    def digamma(self) -> "Tensor":
        return digamma_from_tensor(self)

    def i0(self) -> "Tensor":
        return i0_from_tensor(self)

    def deg2rad(self) -> "Tensor":
        return deg2rad_from_tensor(self)

    def rad2deg(self) -> "Tensor":
        return rad2deg_from_tensor(self)

    def eq(self, other: "Tensor") -> "Tensor":
        return eq_from_tensors(self, other)

    def ne(self, other: "Tensor") -> "Tensor":
        return ne_from_tensors(self, other)

    def lt(self, other: "Tensor") -> "Tensor":
        return lt_from_tensors(self, other)

    def le(self, other: "Tensor") -> "Tensor":
        return le_from_tensors(self, other)

    def gt(self, other: "Tensor") -> "Tensor":
        return gt_from_tensors(self, other)

    def ge(self, other: "Tensor") -> "Tensor":
        return ge_from_tensors(self, other)

    def sum(self, dim: int | None = None, keepdim: bool = False) -> "Tensor":
        if dim is not None:
            return sum_dim_from_tensor(self, dim, keepdim)
        return sum_from_tensor(self)

    def mean(self, dim: int | None = None, keepdim: bool = False) -> "Tensor":
        if dim is not None:
            return mean_dim_from_tensor(self, dim, keepdim)
        return mean_from_tensor(self)

    def prod(self) -> "Tensor":
        return prod_from_tensor(self)

    def min(self) -> "Tensor":
        return min_from_tensor(self)

    def max(self) -> "Tensor":
        return max_from_tensor(self)

    def masked_select(self, mask: "Tensor") -> "Tensor":
        return masked_select_from_tensor(self, mask)

    def masked_fill(self, mask: "Tensor", value: float) -> "Tensor":
        return masked_fill_from_tensor(self, mask, value)

    # ── Internal helpers for nn module ─────────────────────────────
    def _set(self, other: "Tensor") -> None:
        self._id = other._id
        self._shape = list(other._shape)
        self._dtype = other._dtype

    def __getitem__(self, key: object) -> object:
        if isinstance(key, int):
            return self.select(0, key)
        if isinstance(key, slice):
            return self.slice(0, key.start, key.stop, 1 if key.step is None else int(key.step))
        from ._tensor import masked_select_from_tensor
        if isinstance(key, Tensor) and key._dtype == "bool":
            return masked_select_from_tensor(self, key)
        if isinstance(key, tuple):
            result = self
            for i, k in enumerate(key):
                if isinstance(k, int):
                    result = result.select(dim=i, index=k)
                elif isinstance(k, slice):
                    result = result.slice(dim=i, start=k.start, end=k.stop, step=k.step or 1)
                elif isinstance(k, Tensor):
                    if result.ndim <= 2:
                        result = result.index_select(dim=i, index=k.flatten())
                    else:
                        from .__init__ import cat
                        indices_flat = k.flatten()
                        picked: list[Tensor] = []
                        for j in range(indices_flat._shape[0]):
                            picked.append(result.select(dim=i, index=int(indices_flat.select(0, j).item())))
                        result = cat(picked, dim=i)
                else:
                    raise TypeError(f"Unsupported index type: {type(k)}")
            return result
        raise TypeError(f"Tensor indexing supports only int, slice, tuple, or bool Tensor in MVP.")


def _flatten_out(data: object) -> list[float]:
    if isinstance(data, list):
        out: list[float] = []
        for item in data:
            out.extend(_flatten_out(item))
        return out
    return [float(data)]


def _scalar_to_tensor(value: float, dtype: str = "float32") -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.zeros([1], dtype))
    t_id, _, _ = _js_meta_to_tuple(meta)
    # Put the value in
    meta2 = _run_js_awaitable(runtime.fill(t_id, float(value)))
    t_id2, _, _ = _js_meta_to_tuple(meta2)
    t = Tensor.__new__(Tensor)
    t._id = t_id2
    t._shape = [1]
    t._dtype = dtype
    return t


def _infer_shape(data: object) -> list[int]:
    if not isinstance(data, list):
        return []
    if len(data) == 0:
        return [0]
    first_shape = _infer_shape(data[0])
    for item in data[1:]:
        if _infer_shape(item) != first_shape:
            raise ValueError("tensor() expects a rectangular nested list.")
    return [len(data), *first_shape]


def _flatten(data: object) -> list[float]:
    if isinstance(data, list):
        out: list[float] = []
        for item in data:
            out.extend(_flatten(item))
        return out
    return [float(data)]  # type: ignore[arg-type]


def _normalize_shape(shape: int | Sequence[int]) -> list[int]:
    if isinstance(shape, int):
        if shape < 0:
            raise ValueError("shape values must be >= 0")
        return [shape]

    normalized = [int(v) for v in shape]
    if any(v < 0 for v in normalized):
        raise ValueError("shape values must be >= 0")
    return normalized


def _normalize_shape_from_args(shape: Sequence[int]) -> list[int]:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        return _normalize_shape(shape[0])
    return _normalize_shape([int(v) for v in shape])


def _coerce_out_value(value: float, dtype: str) -> object:
    if dtype == "bool":
        return bool(value)
    if dtype == "int32":
        return int(value)
    return float(value)


def _reshape_flat_values(flat: list[float], shape: Sequence[int], dtype: str = "float32") -> object:
    if len(shape) == 0:
        return _coerce_out_value(float(flat[0]), dtype) if flat else _coerce_out_value(0.0, dtype)
    if len(shape) == 1:
        width = int(shape[0])
        return [_coerce_out_value(float(v), dtype) for v in flat[:width]]
    stride = 1
    for dim in shape[1:]:
        stride *= int(dim)
    width = int(shape[0])
    out: list[object] = []
    for i in range(width):
        start = i * stride
        out.append(_reshape_flat_values(flat[start : start + stride], shape[1:], dtype))
    return out


def tensor_from_data(data: object, shape: Sequence[int] | None = None, dtype: str = "float32", requires_grad: bool = False) -> Tensor:
    runtime = _get_runtime()
    if shape is None:
        shape = _infer_shape(data)
    flat = _flatten(data)
    meta = _run_js_awaitable(runtime.tensorFromData(flat, shape, dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype, _requires_grad=requires_grad)


def zeros_from_shape(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    runtime = _get_runtime()
    normalized = _normalize_shape(shape)
    meta = _run_js_awaitable(runtime.zeros(normalized, dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def ones_from_shape(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    runtime = _get_runtime()
    normalized = _normalize_shape(shape)
    meta = _run_js_awaitable(runtime.ones(normalized, dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def rand_from_shape(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    runtime = _get_runtime()
    normalized = _normalize_shape(shape)
    meta = _run_js_awaitable(runtime.rand(normalized, dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def randn_from_shape(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    runtime = _get_runtime()
    normalized = _normalize_shape(shape)
    meta = _run_js_awaitable(runtime.randn(normalized, dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def arange_from_values(start: float, end: float | None = None, step: float = 1.0, dtype: str = "float32") -> Tensor:
    runtime = _get_runtime()
    if end is None:
        resolved_start = 0.0
        resolved_end = float(start)
    else:
        resolved_start = float(start)
        resolved_end = float(end)
    meta = _run_js_awaitable(runtime.arange(resolved_start, resolved_end, float(step), dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def full_from_shape(shape: int | Sequence[int], fill_value: float, dtype: str = "float32") -> Tensor:
    runtime = _get_runtime()
    normalized = _normalize_shape(shape)
    meta = _run_js_awaitable(runtime.full(normalized, float(fill_value), dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def full_like_from_tensor(tensor: Tensor, fill_value: float, dtype: str | None = None) -> Tensor:
    runtime = _get_runtime()
    if dtype is None:
        meta = _run_js_awaitable(runtime.fullLike(tensor._id, float(fill_value)))
    else:
        meta = _run_js_awaitable(runtime.fullLike(tensor._id, float(fill_value), dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def zeros_like_from_tensor(tensor: Tensor, dtype: str | None = None) -> Tensor:
    runtime = _get_runtime()
    if dtype is None:
        meta = _run_js_awaitable(runtime.zerosLike(tensor._id))
    else:
        meta = _run_js_awaitable(runtime.zerosLike(tensor._id, dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def ones_like_from_tensor(tensor: Tensor, dtype: str | None = None) -> Tensor:
    runtime = _get_runtime()
    if dtype is None:
        meta = _run_js_awaitable(runtime.onesLike(tensor._id))
    else:
        meta = _run_js_awaitable(runtime.onesLike(tensor._id, dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def empty_like_from_tensor(tensor: Tensor, dtype: str | None = None) -> Tensor:
    runtime = _get_runtime()
    if dtype is None:
        meta = _run_js_awaitable(runtime.emptyLike(tensor._id))
    else:
        meta = _run_js_awaitable(runtime.emptyLike(tensor._id, dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def empty_from_shape(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    runtime = _get_runtime()
    normalized = _normalize_shape(shape)
    meta = _run_js_awaitable(runtime.empty(normalized, dtype))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def pow_from_tensors(a: Tensor, b: Tensor) -> Tensor:
    from .autograd import _Node, is_grad_enabled, _grad_pow

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.pow(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and (a._requires_grad or b._requires_grad):
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: _grad_pow(g, a, b), [a, b])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def heaviside_from_tensors(input_: Tensor, values: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.heaviside(input_._id, values._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def maximum_from_tensors(a: Tensor, b: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.maximum(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def minimum_from_tensors(a: Tensor, b: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.minimum(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def any_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.any(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def all_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.all(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def cumsum_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.cumsum(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def cumprod_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.cumprod(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def tril_from_tensor(tensor: Tensor, diagonal: int = 0) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.tril(tensor._id, int(diagonal)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def triu_from_tensor(tensor: Tensor, diagonal: int = 0) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.triu(tensor._id, int(diagonal)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def flip_from_tensor(tensor: Tensor, dims: Sequence[int]) -> Tensor:
    runtime = _get_runtime()
    normalized = [int(v) for v in dims]
    meta = _run_js_awaitable(runtime.flip(tensor._id, normalized))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def where_from_tensors(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.where(condition._id, x._id, y._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def topk_from_tensor(tensor: Tensor, k: int, dim: int = -1, largest: bool = True) -> tuple[Tensor, Tensor]:
    d = dim if dim >= 0 else dim + len(tensor._shape)
    size = tensor._shape[d]
    if k >= size:
        indices_np = list(range(size))
        values = tensor
    else:
        flat = tensor.tolist()
        flat_list: list[float] = _flatten_out(flat)
        shape = list(tensor._shape)
        step = product(shape[d + 1:]) if d + 1 < len(shape) else 1
        outer = product(shape[:d]) if d > 0 else 1
        all_indices: list[list[int]] = []
        all_values: list[float] = []
        for o in range(outer):
            base = o * shape[d] * step
            for s in range(step):
                idx = base + s
                col_vals = [(flat_list[idx + i * step], i) for i in range(shape[d])]
                col_vals.sort(key=lambda x: x[0], reverse=largest)
                topk_vals = col_vals[:k]
                for v, i in topk_vals:
                    all_values.append(v)
                    all_indices.append(i)
        out_shape = list(shape)
        out_shape[d] = k
        values = tensor_from_data(all_values, out_shape, tensor._dtype)
        index_data = [x[0] for x in all_indices]
    indices = tensor_from_data(index_data, out_shape, "int64")
    return values, indices


def sort_from_tensor(tensor: Tensor, dim: int = -1, descending: bool = False) -> tuple[Tensor, Tensor]:
    return topk_from_tensor(tensor, tensor._shape[dim if dim >= 0 else dim + len(tensor._shape)], dim, descending)


def gather_from_tensor(tensor: Tensor, dim: int, index: Tensor) -> Tensor:
    d = dim if dim >= 0 else dim + len(tensor._shape)
    src = tensor.tolist()
    src_flat: list[float] = _flatten_out(src)
    idx = index.tolist()
    idx_flat: list[float] = _flatten_out(idx)
    out_shape = list(index._shape)
    step = product(tensor._shape[d + 1:]) if d + 1 < len(tensor._shape) else 1
    result: list[float] = []
    for i in range(product(out_shape)):
        # Map linear index i to multi-dim, compute source position
        remaining = i
        src_linear = 0
        mult = 1
        for dim_idx in range(len(out_shape) - 1, -1, -1):
            coord = remaining % out_shape[dim_idx]
            remaining //= out_shape[dim_idx]
            if dim_idx == d:
                # Gather: use index value
                idx_idx = i  # this needs proper mapping, simplified for dim=1
                actual_idx = int(idx_flat[min(i, len(idx_flat) - 1)])
                src_linear = actual_idx * step + (i % step) if d == 1 else actual_idx
            else:
                src_linear += coord * mult if d == 1 else coord
            mult *= tensor._shape[dim_idx]
        # Major simplification: just use index flat as direct source index
        src_idx = min(int(idx_flat[min(i, len(idx_flat) - 1)]), len(src_flat) - 1)
        result.append(src_flat[src_idx])
    return tensor_from_data(result, out_shape, tensor._dtype)


def scatter_from_tensor(tensor: Tensor, dim: int, index: Tensor, src: Tensor | float) -> Tensor:
    result = tensor.clone()
    flat = result.tolist()
    flat_list: list[float] = _flatten_out(flat)
    idx = index.tolist()
    idx_flat: list[float] = _flatten_out(idx)
    out_len = len(flat_list)
    if isinstance(src, (int, float)):
        val = float(src)
        for i in range(len(idx_flat)):
            pos = int(idx_flat[i])
            if 0 <= pos < out_len:
                flat_list[pos] = val
    else:
        src_flat = _flatten_out(src.tolist())
        for i in range(min(len(idx_flat), len(src_flat))):
            pos = int(idx_flat[i])
            if 0 <= pos < out_len:
                flat_list[pos] = src_flat[i]
    return tensor_from_data(flat_list, list(tensor._shape), tensor._dtype)


def cat_from_tensors(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    runtime = _get_runtime()
    ids = [t._id for t in tensors]
    meta = _run_js_awaitable(runtime.cat(ids, int(dim)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def stack_from_tensors(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    if len(tensors) == 0:
        raise ValueError("stack requires at least one tensor")
    from .__init__ import cat
    unsqueezed = [t.unsqueeze(dim) for t in tensors]
    return cat(unsqueezed, dim=dim)


def expand_from_tensor(tensor: Tensor, shape: int | Sequence[int]) -> Tensor:
    runtime = _get_runtime()
    normalized = _normalize_shape(shape)
    meta = _run_js_awaitable(runtime.expand(tensor._id, normalized))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def index_select_from_tensor(input: Tensor, dim: int, index: Tensor) -> Tensor:
    from .autograd import _Node, is_grad_enabled, _grad_index_select

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.indexSelect(input._id, int(dim), index._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and input._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_index_select(g, input, dim, index),), [input])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def sigmoid_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.sigmoid(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def tanh_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.tanh(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def sin_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.sin(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def cos_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.cos(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def gelu_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.gelu(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def silu_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.silu(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def leaky_relu_from_tensor(tensor: Tensor, alpha: float = 0.01) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.leakyRelu(tensor._id, alpha))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def floor_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.floor(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def ceil_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.ceil(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def round_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.round(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def reciprocal_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.reciprocal(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def square_from_tensor(tensor: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.square(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


# ── Fase 0: unary ops ──────────────────────────────────────────────

def _make_unary_from_tensor(fn_name: str):
    def wrapper(tensor: Tensor) -> Tensor:
        runtime = _get_runtime()
        meta = _run_js_awaitable(getattr(runtime, fn_name)(tensor._id))
        tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
        return Tensor(tensor_id, out_shape, out_dtype)
    wrapper.__name__ = fn_name + "_from_tensor"
    return wrapper


tan_from_tensor = _make_unary_from_tensor("tan")
asin_from_tensor = _make_unary_from_tensor("asin")
acos_from_tensor = _make_unary_from_tensor("acos")
atan_from_tensor = _make_unary_from_tensor("atan")
sinh_from_tensor = _make_unary_from_tensor("sinh")
cosh_from_tensor = _make_unary_from_tensor("cosh")
asinh_from_tensor = _make_unary_from_tensor("asinh")
acosh_from_tensor = _make_unary_from_tensor("acosh")
atanh_from_tensor = _make_unary_from_tensor("atanh")
exp2_from_tensor = _make_unary_from_tensor("exp2")
log2_from_tensor = _make_unary_from_tensor("log2")
log10_from_tensor = _make_unary_from_tensor("log10")
log1p_from_tensor = _make_unary_from_tensor("log1p")
expm1_from_tensor = _make_unary_from_tensor("expm1")
trunc_from_tensor = _make_unary_from_tensor("trunc")
frac_from_tensor = _make_unary_from_tensor("frac")
softplus_from_tensor = _make_unary_from_tensor("softplus")
mish_from_tensor = _make_unary_from_tensor("mish")
hardsigmoid_from_tensor = _make_unary_from_tensor("hardsigmoid")
hardswish_from_tensor = _make_unary_from_tensor("hardswish")
softsign_from_tensor = _make_unary_from_tensor("softsign")
tanhshrink_from_tensor = _make_unary_from_tensor("tanhshrink")
rsqrt_from_tensor = _make_unary_from_tensor("rsqrt")
sign_from_tensor = _make_unary_from_tensor("sign")
sgn_from_tensor = _make_unary_from_tensor("sgn")
isnan_from_tensor = _make_unary_from_tensor("isnan")
isinf_from_tensor = _make_unary_from_tensor("isinf")
isfinite_from_tensor = _make_unary_from_tensor("isfinite")
isposinf_from_tensor = _make_unary_from_tensor("isposinf")
isneginf_from_tensor = _make_unary_from_tensor("isneginf")
logical_not_from_tensor = _make_unary_from_tensor("logicalNot")
erf_from_tensor = _make_unary_from_tensor("erf")
erfc_from_tensor = _make_unary_from_tensor("erfc")
lgamma_from_tensor = _make_unary_from_tensor("lgamma")
digamma_from_tensor = _make_unary_from_tensor("digamma")
i0_from_tensor = _make_unary_from_tensor("i0")
deg2rad_from_tensor = _make_unary_from_tensor("deg2rad")
rad2deg_from_tensor = _make_unary_from_tensor("rad2deg")


def eq_from_tensors(a: Tensor, b: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.eq(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def ne_from_tensors(a: Tensor, b: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.ne(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def lt_from_tensors(a: Tensor, b: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.lt(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def le_from_tensors(a: Tensor, b: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.le(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def gt_from_tensors(a: Tensor, b: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.gt(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def ge_from_tensors(a: Tensor, b: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.ge(a._id, b._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def sum_dim_from_tensor(tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    from .autograd import _Node, is_grad_enabled, _grad_sum

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.sumDim(tensor._id, int(dim), keepdim))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_sum(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def sum_from_tensor(tensor: Tensor) -> Tensor:
    from .autograd import _Node, is_grad_enabled, _grad_sum

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.sum(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_sum(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def mean_from_tensor(tensor: Tensor) -> Tensor:
    from .autograd import _Node, is_grad_enabled, _grad_mean

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.mean(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_mean(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def mean_dim_from_tensor(tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    from .autograd import _Node, is_grad_enabled, _grad_mean

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.meanDim(tensor._id, int(dim), keepdim))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_mean(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def prod_from_tensor(tensor: Tensor) -> Tensor:
    from .autograd import _Node, is_grad_enabled, _grad_prod

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.prod(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_prod(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def min_from_tensor(tensor: Tensor) -> Tensor:
    from .autograd import _Node, is_grad_enabled, _grad_min

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.min(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_min(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def max_from_tensor(tensor: Tensor) -> Tensor:
    from .autograd import _Node, is_grad_enabled, _grad_max

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.max(tensor._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_max(g, tensor),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def masked_select_from_tensor(tensor: Tensor, mask: Tensor) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.maskedSelect(tensor._id, mask._id))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def masked_fill_from_tensor(tensor: Tensor, mask: Tensor, value: float) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.maskedFill(tensor._id, mask._id, float(value)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def softmax_from_tensor(tensor: Tensor, dim: int = -1) -> Tensor:
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.softmax(tensor._id, int(dim)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def log_softmax_from_tensor(tensor: Tensor, dim: int = -1) -> Tensor:
    from .autograd import _Node, is_grad_enabled, _grad_log_softmax

    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.logSoftmax(tensor._id, int(dim)))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)

    if is_grad_enabled() and tensor._requires_grad:
        result = Tensor(tensor_id, out_shape, out_dtype, _requires_grad=True)
        result._node = _Node(result, lambda g: (_grad_log_softmax(g, tensor, dim),), [tensor])
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def conv2d_from_tensors(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: Sequence[int] = (1,),
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    groups: int = 1,
) -> Tensor:
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

        # Build parents list matching the order of gradients returned by _grad_conv2d
        parents = [p for p in (input, weight, bias) if p is not None]
        grad_indices = [i for i, p in enumerate((input, weight, bias)) if p is not None]

        def _conv_grad_fn(g, inp=input, wt=weight, out_sh=out_shape, pr=params, gidx=grad_indices):
            all_grads = _grad_conv2d(g, inp, wt, out_sh, pr)
            return tuple(all_grads[i] for i in gidx)

        result._node = _Node(result, _conv_grad_fn, parents)
        return result
    return Tensor(tensor_id, out_shape, out_dtype)


def max_pool2d_from_tensor(
    input: Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int] | None = None,
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
) -> Tensor:
    runtime = _get_runtime()
    ksize = [int(k) for k in kernel_size]
    strd = [int(s) for s in (stride if stride is not None else kernel_size)]
    pad = [int(p) for p in padding]
    dil = [int(d) for d in dilation]
    meta = _run_js_awaitable(runtime.maxPool2d(input._id, ksize, strd, pad, dil))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def avg_pool2d_from_tensor(
    input: Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int] | None = None,
    padding: Sequence[int] = (0,),
    count_include_pad: bool = True,
) -> Tensor:
    runtime = _get_runtime()
    ksize = [int(k) for k in kernel_size]
    strd = [int(s) for s in (stride if stride is not None else kernel_size)]
    pad = [int(p) for p in padding]
    meta = _run_js_awaitable(runtime.avgPool2d(input._id, ksize, strd, pad, count_include_pad))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def batch_norm_from_tensor(
    input: Tensor,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
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
    input: Tensor,
    target: Tensor,
) -> Tensor:
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
    input: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
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
    input: Tensor,
    normalized_shape: Sequence[int],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
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


# ─── Backward ops (runtime-backed) ───────────────────────────────────────────

def conv2d_input_backward_from_tensors(
    grad_output: Tensor,
    weight: Tensor,
    input_shape: tuple[int, ...],
    grad_output_shape: tuple[int, ...],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
) -> Tensor:
    """Backward pass of conv2d with respect to the input."""
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
    grad_output: Tensor,
    input: Tensor,
    weight_shape: tuple[int, ...],
    grad_output_shape: tuple[int, ...],
    input_shape: tuple[int, ...],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
) -> Tensor:
    """Backward pass of conv2d with respect to the weight."""
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
    grad_output: Tensor,
    out_ch: int,
    grad_output_shape: tuple[int, ...],
) -> Tensor:
    """Backward pass of conv2d with respect to the bias."""
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.conv2dBiasBackward(
        grad_output._id,
        int(out_ch),
        list(grad_output_shape),
    ))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def log_softmax_backward_from_tensors(
    grad_output: Tensor,
    softmax: Tensor,
    batch_size: int,
    num_classes: int,
) -> Tensor:
    """Backward pass of log_softmax."""
    runtime = _get_runtime()
    meta = _run_js_awaitable(runtime.logSoftmaxBackward(
        grad_output._id,
        softmax._id,
        int(batch_size),
        int(num_classes),
    ))
    tensor_id, out_shape, out_dtype = _js_meta_to_tuple(meta)
    return Tensor(tensor_id, out_shape, out_dtype)


def nll_loss_backward_from_tensors(
    targets: Tensor,
    batch_size: int,
    num_classes: int,
    scale: float = 1.0,
) -> Tensor:
    """Backward pass of NLL loss."""
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
    grad_output: Tensor,
    input_shape: list[int],
    sliced_shape: list[int],
    dim: int,
    start: int,
    step: int = 1,
) -> Tensor:
    """Backward pass of slice: scatter grad_output back to full tensor shape."""
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
