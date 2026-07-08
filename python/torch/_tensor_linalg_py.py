from __future__ import annotations

from typing import TYPE_CHECKING

from ._runtime import _get_runtime, _run_js_awaitable

if TYPE_CHECKING:
    from ._tensor import Tensor


def det_from_tensor(tensor: "Tensor") -> "Tensor":
    from .tensor_factories_ops import tensor_from_data
    from ._api_creation import arange

    n = tensor._shape[-1]
    a_lu, pivot = tensor.lu()
    # Compute sign from pivot parity (small CPU readback: n integers, unavoidable).
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
    # Compute product of U diagonal on GPU via linear indices -> gather -> prod.
    flat_indices = arange(0, n, 1, dtype="int64").mul(n + 1)
    diag_vals = a_lu.reshape(-1).index_select(0, flat_indices)
    return diag_vals.prod().mul(sign)


def inv_from_tensor(tensor: "Tensor") -> "Tensor":
    from .__init__ import cat, tril, triu, eye

    n = tensor._shape[-1]
    a_lu, _pivot = tensor.lu()
    l_part = tril(a_lu, diagonal=-1)
    l_full = eye(n, dtype=tensor.dtype) + l_part
    u_full = triu(a_lu, diagonal=0)
    inv_cols = []
    for j in range(n):
        col = eye(n, dtype=tensor.dtype).select(1, j).reshape([n, 1])
        y = l_full.triangular_solve(col, upper=False)
        x = u_full.triangular_solve(y, upper=True)
        inv_cols.append(x)
    return cat(inv_cols, dim=1)


def diag_from_tensor(tensor: "Tensor") -> "Tensor":
    from .tensor_factories_ops import tensor_from_data
    from ._api_creation import arange, zeros, eye, full
    from .__init__ import cat

    if len(tensor._shape) == 1:
        n = tensor._shape[0]
        eye_m = eye(n, dtype=tensor._dtype)
        return eye_m.mul(tensor.reshape([1, n]))
    n = tensor._shape[-1]
    nrows = tensor._shape[0]
    k = min(nrows, n)
    flat_indices = arange(0, k, 1, dtype="int64").mul(n + 1)
    return tensor.reshape(-1).index_select(0, flat_indices)
