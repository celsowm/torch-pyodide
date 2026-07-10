from __future__ import annotations

from typing import Sequence
import math

import torch
from torch import Tensor
from torch.tensor_factories_ops import tensor_from_data
from torch.tensor_shape_utils import _flatten, _infer_shape


def _eigh_jacobi_cpu(x: Tensor) -> tuple[Tensor, Tensor]:
    """Eigen-decomposition for symmetric matrices via Jacobi iteration (CPU)."""
    n = x.shape[-1]
    A = x.tolist()
    A_flat = _flatten(A)
    A_mat = [A_flat[i * n:(i + 1) * n] for i in range(n)]

    V_mat = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _iter in range(50):
        max_off = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                val = abs(A_mat[i][j])
                if val > max_off:
                    max_off = val
                    p, q = i, j
        if max_off < 1e-12:
            break
        if A_mat[p][p] == A_mat[q][q]:
            theta = math.pi / 4.0
        else:
            theta = 0.5 * math.atan2(2.0 * A_mat[p][q], A_mat[p][p] - A_mat[q][q])
        c = math.cos(theta)
        s = math.sin(theta)
        app = A_mat[p][p]
        aqq = A_mat[q][q]
        apq = A_mat[p][q]
        A_mat[p][p] = c * c * app + s * s * aqq - 2.0 * s * c * apq
        A_mat[q][q] = s * s * app + c * c * aqq + 2.0 * s * c * apq
        A_mat[p][q] = 0.0
        A_mat[q][p] = 0.0
        for r in range(n):
            if r != p and r != q:
                apr = A_mat[p][r]
                aqr = A_mat[q][r]
                A_mat[p][r] = c * apr - s * aqr
                A_mat[r][p] = A_mat[p][r]
                A_mat[q][r] = s * apr + c * aqr
                A_mat[r][q] = A_mat[q][r]
        for r in range(n):
            vrp = V_mat[r][p]
            vrq = V_mat[r][q]
            V_mat[r][p] = c * vrp - s * vrq
            V_mat[r][q] = s * vrp + c * vrq

    eigenvalues = [A_mat[i][i] for i in range(n)]
    eigenvectors_flat: list[float] = []
    for i in range(n):
        for j in range(n):
            eigenvectors_flat.append(V_mat[j][i])
    ev = tensor_from_data(eigenvalues, [n], x.dtype)
    evect = tensor_from_data(eigenvectors_flat, [n, n], x.dtype)
    return ev, evect


def _eigh_jacobi(x: Tensor) -> tuple[Tensor, Tensor]:
    """Eigen-decomposition for symmetric matrices via GPU-backed Jacobi iteration.

    Each Jacobi rotation is a single WebGPU kernel (``jacobi.wgsl``); the Python
    side keeps only the control flow (picking the rotation and computing ``c,s``
    from a few scalar reads), so no ``.tolist()`` of the full matrix is needed.
    """
    try:
        from .._tensor_runtime_bridge import jacobi_rotate_from_tensors

        n = x.shape[-1]
        A = x
        V = torch.eye(n, dtype=x.dtype)
        eye_mask = 1.0 - torch.eye(n, dtype=x.dtype)
        eps = 1e-10
        for _sweep in range(100):
            max_off = float((A * eye_mask).abs().max().item())
            if max_off < eps:
                break
            diag_list = A.diag().tolist()
            for i in range(n):
                for j in range(i + 1, n):
                    apq = float(A[i, j].item())
                    if abs(apq) < eps:
                        continue
                    ap = diag_list[i]
                    aq = diag_list[j]
                    theta = 0.5 * math.atan2(2.0 * apq, aq - ap)
                    c = math.cos(theta)
                    s = math.sin(theta)
                    A, V = jacobi_rotate_from_tensors(A, V, i, j, c, s)
        eigvals = A.diag()
        return eigvals, V
    except Exception:
        return _eigh_jacobi_cpu(x)


def svd(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    from ..tensor_ops import reciprocal_from_tensor

    m, n = x.shape
    if m >= n:
        xtx = x.permute([1, 0]).matmul(x)
        eigvals, V = _eigh_jacobi(xtx)
        idx = eigvals.argsort(descending=True)
        eigvals = eigvals.index_select(0, idx)
        V = V.index_select(1, idx)
        s = eigvals.clamp(0.0).sqrt()
        S_inv = reciprocal_from_tensor(s.clamp(min=1e-12)).reshape(1, n)
        U = x.matmul(V).mul(S_inv)
        return U, s, V.permute([1, 0])
    else:
        xxt = x.matmul(x.permute([1, 0]))
        eigvals, U = _eigh_jacobi(xxt)
        idx = eigvals.argsort(descending=True)
        eigvals = eigvals.index_select(0, idx)
        U = U.index_select(1, idx)
        s = eigvals.clamp(0.0).sqrt()
        S_inv = reciprocal_from_tensor(s.clamp(min=1e-12)).reshape(m, 1)
        V = x.permute([1, 0]).matmul(U).mul(S_inv)
        return U, s, V.permute([1, 0])


def qr(x: Tensor, mode: str = "reduced") -> tuple[Tensor, Tensor]:
    m, n = x.shape
    cols: list[Tensor] = []
    R_list: list[list[float]] = [[0.0] * n for _ in range(n)]
    # Gram-Schmidt on the GPU: each column op (dot, subtract, norm) is a GPU op,
    # removing the previous pure-Python loop over x.tolist().
    for j in range(n):
        v = x.select(1, j)
        for i in range(j):
            qi = cols[i]
            dot = (qi * v).sum()
            R_list[i][j] = float(dot)
            v = v - qi * dot
        norm = float((v * v).sum().sqrt())
        R_list[j][j] = norm
        if norm > 1e-12:
            qj = v / norm
        else:
            qj = v * 0.0
        cols.append(qj)
    from ..tensor_ops import cat_multi_from_tensors

    Q = cat_multi_from_tensors([c.unsqueeze(1) for c in cols], dim=1)
    R = tensor_from_data(
        [R_list[i][j] for i in range(n) for j in range(n)], [n, n], x.dtype
    )
    return Q, R


def eigh(x: Tensor) -> tuple[Tensor, Tensor]:
    return _eigh_jacobi(x)


def eig(x: Tensor) -> tuple[Tensor, Tensor]:
    return _eigh_jacobi(x)


def solve(A: Tensor, B: Tensor) -> Tensor:
    n = A.shape[-1]
    A_lu, pivot = A.lu()
    l_part = torch.tril(A_lu, diagonal=-1)
    l_full = torch.eye(n, dtype=A.dtype) + l_part
    u_full = torch.triu(A_lu, diagonal=0)
    if len(B.shape) == 1:
        col = B.reshape([n, 1])
        y = l_full.triangular_solve(col, upper=False)
        return u_full.triangular_solve(y, upper=True).reshape([n])
    m = B.shape[-1]
    result_cols = []
    for j in range(m):
        col = B.select(dim=1, index=j).reshape([n, 1])
        y = l_full.triangular_solve(col, upper=False)
        x = u_full.triangular_solve(y, upper=True)
        result_cols.append(x)
    return torch.cat(result_cols, dim=1)


def pinv(x: Tensor, rcond: float = 1e-15) -> Tensor:
    from ..tensor_ops import reciprocal_from_tensor, where_from_tensors, zeros_from_shape

    U, s, Vt = svd(x)
    smax = float(s.max().item())
    tol = rcond * smax if smax > 0 else rcond
    s_inv = where_from_tensors(
        s > tol,
        reciprocal_from_tensor(s.clamp(min=1e-12)),
        zeros_from_shape(s._shape, s._dtype),
    )
    return Vt.permute([1, 0]).matmul(torch.diag(s_inv)).matmul(U.permute([1, 0]))


def matrix_power(x: Tensor, n: int) -> Tensor:
    if n == 0:
        return torch.eye(x.shape[-1], dtype=x.dtype)
    if n < 0:
        x = inv(x)
        n = -n
    result = x
    for _ in range(n - 1):
        result = result.matmul(x)
    return result


def matrix_rank(x: Tensor, tol: float | None = None) -> Tensor:
    _, s, _ = svd(x)
    sv_list = s.tolist()
    if tol is None:
        tol = max(sv_list) * max(x.shape) * 1e-7
    rank = sum(1.0 for sv in sv_list if sv > tol)
    return tensor_from_data([rank], x.dtype)


def norm(x: Tensor, ord: float | str | None = None) -> Tensor:
    return x.norm(p=ord if ord is not None else "fro")


def det(x: Tensor) -> Tensor:
    return x.det()


def inv(x: Tensor) -> Tensor:
    return x.inv()


def cholesky(x: Tensor) -> Tensor:
    return x.cholesky()


def lu(x: Tensor) -> tuple[Tensor, Tensor]:
    return x.lu()


def lstsq(A: Tensor, B: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    X = pinv(A).matmul(B)
    residual = (A.matmul(X) - B).norm()
    rank = matrix_rank(A)
    _, S, _ = svd(A)
    return X, residual, rank, S


__all__ = [
    "svd", "qr", "eig", "eigh", "solve", "pinv", "inv",
    "det", "cholesky", "lu", "matrix_power", "matrix_rank",
    "norm", "lstsq",
]
