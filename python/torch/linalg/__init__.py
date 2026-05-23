from __future__ import annotations

from typing import Sequence
import math

import torch
from torch import Tensor
from torch.tensor_factories_ops import tensor_from_data
from torch.tensor_shape_utils import _flatten, _infer_shape


def _eigh_jacobi(x: Tensor) -> tuple[Tensor, Tensor]:
    """Eigen-decomposition for symmetric matrices via Jacobi iteration."""
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


def svd(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    m, n = x.shape
    if m >= n:
        xtx = x.T.matmul(x)
        eigvals, V = _eigh_jacobi(xtx)
        idx = eigvals.argsort(descending=True)
        eigvals = eigvals.gather(0, idx)
        V = V.gather(1, idx)
        s = eigvals.clamp(0.0).sqrt()
        S_inv = tensor_from_data([0.0 if sv == 0.0 else 1.0 / sv for sv in s.tolist()], s.dtype).reshape(1, n)
        U = x.matmul(V).mul(S_inv)
        return U, s, V.T
    else:
        xxt = x.matmul(x.T)
        eigvals, U = _eigh_jacobi(xxt)
        idx = eigvals.argsort(descending=True)
        eigvals = eigvals.gather(0, idx)
        U = U.gather(1, idx)
        s = eigvals.clamp(0.0).sqrt()
        S_inv = tensor_from_data([0.0 if sv == 0.0 else 1.0 / sv for sv in s.tolist()], s.dtype).reshape(m, 1)
        V = x.T.matmul(U).mul(S_inv)
        return U, s, V.T


def qr(x: Tensor, mode: str = "reduced") -> tuple[Tensor, Tensor]:
    m, n = x.shape
    A = x.tolist()
    A_flat = _flatten(A)
    Q_list: list[float] = [0.0] * (m * n)
    R_list: list[float] = [0.0] * (n * n)
    for j in range(n):
        v = [A_flat[i * n + j] for i in range(m)]
        for i in range(j):
            dot = 0.0
            for k in range(m):
                dot += Q_list[k * n + i] * A_flat[k * n + j]
            R_list[i * n + j] = dot
            for k in range(m):
                v[k] -= dot * Q_list[k * n + i]
        norm = math.sqrt(sum(vk * vk for vk in v))
        R_list[j * n + j] = norm
        if norm > 1e-12:
            for k in range(m):
                Q_list[k * n + j] = v[k] / norm
        else:
            for k in range(m):
                Q_list[k * n + j] = 0.0
    Q = tensor_from_data(Q_list, [m, n], x.dtype)
    R = tensor_from_data(R_list, [n, n], x.dtype)
    return Q, R


def eigh(x: Tensor) -> tuple[Tensor, Tensor]:
    return _eigh_jacobi(x)


def eig(x: Tensor) -> tuple[Tensor, Tensor]:
    return _eigh_jacobi(x)


def solve(A: Tensor, B: Tensor) -> Tensor:
    n = A.shape[-1]
    A_lu, pivot = A.lu()
    l_part = torch.tril(A_lu, diagonal=-1)
    eye_data = [0.0] * (n * n)
    for i in range(n):
        eye_data[i * n + i] = 1.0
    l_full = tensor_from_data(eye_data, A.dtype).reshape([n, n]) + l_part
    u_full = torch.triu(A_lu, diagonal=0)
    m = B.shape[-1] if len(B.shape) > 1 else 1
    col_list = B.tolist()
    col_flat = _flatten(col_list)
    result_cols = []
    for j in range(m):
        col = tensor_from_data([col_flat[i * m + j] for i in range(n)], A.dtype).reshape([n, 1])
        y = l_full.triangular_solve(col, upper=False)
        x = u_full.triangular_solve(y, upper=True)
        result_cols.append(x)
    return torch.cat(result_cols, dim=1)


def pinv(x: Tensor, rcond: float = 1e-15) -> Tensor:
    U, s, Vt = svd(x)
    sv_list = s.tolist()
    tol = rcond * max(sv_list) if max(sv_list) > 0 else rcond
    s_inv = tensor_from_data(
        [1.0 / sv if sv > tol else 0.0 for sv in sv_list],
        s.dtype
    )
    return Vt.T.matmul(torch.diag(s_inv)).matmul(U.T)


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
