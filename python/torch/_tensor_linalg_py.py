from __future__ import annotations

from typing import TYPE_CHECKING

from ._runtime import _get_runtime, _run_js_awaitable

if TYPE_CHECKING:
    from ._tensor import Tensor


def det_from_tensor(tensor: "Tensor") -> "Tensor":
    from .tensor_factories_ops import tensor_from_data

    n = tensor._shape[-1]
    a_lu, pivot = tensor.lu()
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
    return tensor_from_data(sign * u_diag_prod, tensor._dtype)


def inv_from_tensor(tensor: "Tensor") -> "Tensor":
    from .__init__ import cat, tril, triu
    from .tensor_factories_ops import tensor_from_data

    n = tensor._shape[-1]
    a_lu, _pivot = tensor.lu()
    l_part = tril(a_lu, diagonal=-1)
    eye_data = [0.0] * (n * n)
    for i in range(n):
        eye_data[i * n + i] = 1.0
    l_full = tensor_from_data(eye_data, tensor._dtype).reshape([n, n]) + l_part
    u_full = triu(a_lu, diagonal=0)
    inv_cols = []
    for j in range(n):
        col = tensor_from_data([1.0 if i == j else 0.0 for i in range(n)], tensor._dtype).reshape([n, 1])
        y = l_full.triangular_solve(col, upper=False)
        x = u_full.triangular_solve(y, upper=True)
        inv_cols.append(x)
    return cat(inv_cols, dim=1)


def diag_from_tensor(tensor: "Tensor") -> "Tensor":
    from .tensor_factories_ops import tensor_from_data
    from .tensor_shape_utils import _flatten

    flat_list_raw = tensor.tolist()
    if isinstance(flat_list_raw, list):
        flat_list: list[float] = _flatten(flat_list_raw)
    else:
        flat_list = [float(flat_list_raw)]
    if len(tensor._shape) == 1:
        n = tensor._shape[0]
        data: list[float] = [0.0] * (n * n)
        for i in range(n):
            data[i * n + i] = flat_list[i]
        return tensor_from_data(data, [n, n], tensor._dtype)
    n = tensor._shape[-1]
    nrows = tensor._shape[0]
    result_data = [flat_list[i * n + i] for i in range(min(nrows, n))]
    return tensor_from_data(result_data, tensor._dtype)
