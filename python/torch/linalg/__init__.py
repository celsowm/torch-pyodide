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
    LU, pivots = lu_factor(A)
    if len(B.shape) == 1:
        col = B.reshape([n, 1])
        return lu_solve(LU, pivots, col).reshape([n])
    return lu_solve(LU, pivots, B)


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
    if tol is None:
        # Single-scalar readback of the largest singular value is negligible;
        # the count of kept singular values happens on the GPU.
        tol = float(s.max()) * max(x.shape) * 1e-7
    # (s > tol) on GPU, cast to x.dtype, then sum -> scalar tensor (0-dim).
    return (s > tol).to(x.dtype).sum()


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


def cross(input: Tensor, other: Tensor, dim: int = -1) -> Tensor:
    """3D cross product along ``dim`` (size-3 dimension)."""
    a = input.transpose(dim, -1) if dim != -1 else input
    b = other.transpose(dim, -1) if dim != -1 else other
    a0 = a.select(-1, 0)
    a1 = a.select(-1, 1)
    a2 = a.select(-1, 2)
    b0 = b.select(-1, 0)
    b1 = b.select(-1, 1)
    b2 = b.select(-1, 2)
    c0 = a1 * b2 - a2 * b1
    c1 = a2 * b0 - a0 * b2
    c2 = a0 * b1 - a1 * b0
    res = torch.stack([c0, c1, c2], dim=-1)
    if dim != -1:
        res = res.transpose(dim, -1)
    return res


def slogdet(x: Tensor) -> tuple[Tensor, Tensor]:
    """Sign and log-absolute of the determinant (composed via ``det``)."""
    d = det(x)
    sign = torch.sign(d)
    logabsdet = torch.log(torch.abs(d))
    return sign, logabsdet


def svdvals(x: Tensor) -> Tensor:
    """Singular values only."""
    _, s, _ = svd(x)
    return s


def diagonal(x: Tensor, offset: int = 0, dim1: int = -2, dim2: int = -1) -> Tensor:
    nd = x.ndim
    d1 = dim1 % nd
    d2 = dim2 % nd
    other = [i for i in range(nd) if i not in (d1, d2)]
    x = x.permute(other + [d1, d2])
    n = x.shape[-2]
    m = x.shape[-1]
    if offset >= 0:
        s1, s2 = 0, offset
    else:
        s1, s2 = -offset, 0
    length = min(n - s1, m - s2)
    if length <= 0:
        return torch.zeros(x.shape[:-2] + (0,), dtype=x.dtype)
    rows = torch.arange(s1, s1 + length, dtype="int64")
    cols = torch.arange(s2, s2 + length, dtype="int64")
    lead = (1,) * (nd - 2)
    idx_r = rows.reshape(list(lead) + [length, 1])
    xr = torch.gather(x, -2, idx_r)
    idx_c = cols.reshape(list(lead) + [1, length])
    xc = torch.gather(xr, -1, idx_c)
    idx_d = torch.arange(length, dtype="int64").reshape(list(lead) + [length, 1])
    diag = torch.gather(xc, -1, idx_d)
    return diag.squeeze(-1)


def eigvals(x: Tensor) -> Tensor:
    """Eigenvalues (real part; symmetric matrices)."""
    vals, _ = eig(x)
    return vals


def eigvalsh(x: Tensor) -> Tensor:
    """Eigenvalues of a symmetric matrix."""
    vals, _ = eigh(x)
    return vals


def cond(x: Tensor, p: float | str | None = None) -> Tensor:
    if p is None or p in (2, "fro"):
        _, s, _ = svd(x)
        return s.max(dim=-1)[0] / s.min(dim=-1)[0]
    return vector_norm(x, p) * vector_norm(inv(x), p)


def vector_norm(x: Tensor, ord: float | str = 2, dim=None, keepdim: bool = False) -> Tensor:
    dims = list(range(x.ndim)) if dim is None else ([dim] if isinstance(dim, int) else list(dim))

    def red(fn):
        out = x
        for d in dims:
            out = fn(out, d)
        if not keepdim:
            for d in sorted(dims, reverse=True):
                out = out.squeeze(d)
        return out

    if ord == 0:
        return red(lambda t, d: (t != 0).to(x.dtype).sum(dim=d, keepdim=True))
    if ord == float("inf"):
        return red(lambda t, d: t.abs().max(dim=d, keepdim=True)[0])
    if ord == float("-inf"):
        return red(lambda t, d: t.abs().min(dim=d, keepdim=True)[0])
    if ord == 1:
        return red(lambda t, d: t.abs().sum(dim=d, keepdim=True))
    if ord == 2:
        return red(lambda t, d: t.pow(2).sum(dim=d, keepdim=True)).sqrt()
    return red(lambda t, d: t.abs().pow(ord).sum(dim=d, keepdim=True)).pow(1.0 / ord)


def matrix_norm(x: Tensor, ord: float | str = "fro", dim=(-2, -1), keepdim: bool = False) -> Tensor:
    if isinstance(dim, int):
        d1, d2 = dim, dim + 1
    else:
        d1, d2 = dim[0], dim[1]
    if ord == "fro":
        return vector_norm(x, 2, dim=(d1, d2), keepdim=keepdim)
    if ord == "nuc":
        _, s, _ = svd(x)
        return s.sum(dim=-1, keepdim=keepdim)
    if ord == 2:
        _, s, _ = svd(x)
        return s.max(dim=-1, keepdim=keepdim)[0]
    if ord == -2:
        _, s, _ = svd(x)
        return s.min(dim=-1, keepdim=keepdim)[0]
    if ord == float("inf"):
        return vector_norm(x, float("inf"), dim=d2, keepdim=keepdim)
    if ord == float("-inf"):
        return vector_norm(x, float("-inf"), dim=d1, keepdim=keepdim)
    raise ValueError(f"Unsupported ord {ord!r} for matrix_norm")


def solve_triangular(A: Tensor, B: Tensor, upper: bool = False, left: bool = True,
                     unitriangular: bool = False) -> Tensor:
    from .._tensor_runtime_bridge import triangular_solve_from_tensors

    if not left:
        A_t = A.transpose(-1, -2)
        B_t = B.transpose(-1, -2)
        X_t = solve_triangular(A_t, B_t, upper=not upper, unitriangular=unitriangular)
        return X_t.transpose(-1, -2)
    if unitriangular:
        n = A.shape[-1]
        eye = torch.eye(n, dtype=A.dtype)
        A = A * (1.0 - eye) + eye
    return triangular_solve_from_tensors(A, B, upper)


def lu_factor(x: Tensor, pivot: bool = True) -> tuple[Tensor, Tensor]:
    """LU decomposition with partial pivoting (thin wrapper over ``lu``)."""
    return lu(x)


def _apply_pivot(B: Tensor, pivots: Tensor) -> Tensor:
    n = B.shape[-2]
    idx = pivots.to(dtype="int64")  # runtime lu() returns a 0-indexed permutation
    lead = (1,) * (B.ndim - 2)
    idx = idx.reshape(list(lead) + [n, 1])
    idx = idx.expand(list(lead) + [n, B.shape[-1]])
    return torch.gather(B, -2, idx)


def lu_solve(LU: Tensor, pivots: Tensor, B: Tensor) -> Tensor:
    from .._tensor_runtime_bridge import triangular_solve_from_tensors

    n = LU.shape[-1]
    L = torch.tril(LU, diagonal=-1) + torch.eye(n, dtype=LU.dtype)
    U = torch.triu(LU, diagonal=0)
    Bp = _apply_pivot(B, pivots)
    Y = triangular_solve_from_tensors(L, Bp, upper=False)
    X = triangular_solve_from_tensors(U, Y, upper=True)
    return X


def matrix_exp(x: Tensor) -> Tensor:
    """Matrix exponential via scaling-and-squaring with a Taylor series.

    Works for any square matrix; uses only GPU-backed matmul/add/div.
    """
    if x.ndim < 2 or x.shape[-1] != x.shape[-2]:
        raise ValueError("matrix_exp expects square matrices")
    leading = list(x.shape[:-2])
    n = x.shape[-1]
    flat = x.reshape([-1, n, n])
    B = flat.shape[0]
    out: list[Tensor] = []
    for b in range(B):
        A = flat.select(0, b)
        norm = (A.abs().sum(0)).max()
        s = max(0, int(torch.ceil(torch.log2(norm + 1e-12)).item()))
        A1 = A / (2 ** s)
        R = torch.eye(n, dtype=x.dtype) + A1
        term = A1
        for k in range(2, 10):
            term = term.matmul(A1) / float(k)
            R = R + term
        for _ in range(s):
            R = R.matmul(R)
        out.append(R)
    return torch.stack(out, dim=0).reshape(leading + [n, n])


def multi_dot(tensors: list[Tensor]) -> Tensor:
    if len(tensors) < 2:
        raise ValueError("multi_dot expects at least 2 tensors")
    result = tensors[0]
    for t in tensors[1:]:
        result = result.matmul(t)
    return result


def vander(x: Tensor, N: int | None = None, increasing: bool = False) -> Tensor:
    if x.ndim != 1:
        raise ValueError("vander expects a 1D input")
    n = x.shape[0]
    N = n if N is None else N
    cols: list[Tensor] = []
    for i in range(N):
        if i == 0:
            cols.append(torch.ones(n, dtype=x.dtype))
        else:
            cols.append(x ** i)
    M = torch.stack(cols, dim=-1)
    if not increasing:
        M = M.flip([-1])
    return M


class LinAlgError(RuntimeError):
    """Exception raised for linear algebra related errors (mirrors ``torch.linalg``)."""


def _zero_info(shape) -> Tensor:
    return torch.zeros(list(shape), dtype=torch.long)


def _cholesky_info(L: Tensor) -> Tensor:
    """1-based index of first non-positive diagonal entry (0 = success)."""
    d = diagonal(L)
    flat = d.reshape([-1, d.shape[-1]])
    info: list[int] = []
    for row in flat.tolist():
        idx = 0
        for i, v in enumerate(row):
            if v != v:  # NaN diagonal marks the first non-positive-definite pivot
                idx = i + 1
                break
        info.append(idx)
    shape = list(d.shape[:-1]) if d.ndim > 1 else []
    return tensor_from_data(info, shape, torch.long)


def cholesky_ex(A: Tensor, *, upper: bool = False, check_errors: bool = False) -> tuple[Tensor, Tensor]:
    """Cholesky decomposition returning ``(L, info)`` (info is a long tensor)."""
    L = A.cholesky()
    if upper:
        L = L.transpose(-1, -2)
    return L, _cholesky_info(L)


def inv_ex(A: Tensor, *, check_errors: bool = False) -> tuple[Tensor, Tensor]:
    """Matrix inverse returning ``(A_inv, info)``."""
    n = A.shape[-1]
    eye = torch.eye(n, dtype=A.dtype)
    if A.ndim > 2:
        eye = eye.reshape([1] * (A.ndim - 2) + [n, n]).expand(list(A.shape))
    return solve(A, eye), _zero_info(A.shape[:-2] if A.ndim > 2 else [])


def lu_factor_ex(A: Tensor, *, pivot: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    """LU factorization returning ``(LU, pivots, info)``."""
    LU, piv = lu_factor(A, pivot=pivot)
    return LU, piv, _zero_info(A.shape[:-2] if A.ndim > 2 else [])


def solve_ex(A: Tensor, B: Tensor, *, left: bool = True, upper: bool = False,
             transpose: bool = False, adjoint: bool = False,
             check_errors: bool = False) -> tuple[Tensor, Tensor]:
    """Solves ``A X = B`` returning ``(X, info)``."""
    X = solve(A, B)
    return X, _zero_info(B.shape[:-2] if B.ndim > 2 else [])


def ldl_factor(A: Tensor, *, hermitian: bool = False) -> tuple[Tensor, Tensor]:
    """LDLᵀ factorization (symmetric, no pivoting) returning ``(LD, pivots)``.

    ``LD`` packs the unit lower factor ``L`` in its strictly lower triangle and the
    diagonal ``D`` on its diagonal. ``pivots`` is the identity permutation (1-indexed),
    matching the shape/type expected by ``ldl_solve``.
    """
    n = A.shape[-1]
    flat = A.reshape([-1, n, n])
    B = flat.shape[0]
    LDs: list[list[float]] = []
    pivs: list[int] = []
    for b in range(B):
        M = flat.select(0, b).tolist()
        ld = [[0.0] * n for _ in range(n)]
        for j in range(n):
            d = M[j][j]
            for k in range(j):
                d -= ld[j][k] * ld[j][k] * ld[k][k]
            ld[j][j] = d
            for i in range(j + 1, n):
                s = M[i][j]
                for k in range(j):
                    s -= ld[i][k] * ld[j][k] * ld[k][k]
                ld[i][j] = s / d if d != 0 else 0.0
        LDs.append(ld)
        pivs.extend(range(1, n + 1))
    lead = list(A.shape[:-2]) if A.ndim > 2 else []
    LD = tensor_from_data([v for mat in LDs for row in mat for v in row], lead + [n, n], A.dtype)
    piv = tensor_from_data(pivs, lead + [n], torch.long)
    return LD, piv


def ldl_factor_ex(A: Tensor, *, hermitian: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    """LDLᵀ factorization returning ``(LD, pivots, info)``."""
    LD, piv = ldl_factor(A, hermitian=hermitian)
    return LD, piv, _zero_info(A.shape[:-2] if A.ndim > 2 else [])


def ldl_solve(LD: Tensor, pivots: Tensor, B: Tensor) -> Tensor:
    """Solves ``A X = B`` given an ``ldl_factor`` result ``(LD, pivots)``."""
    from .._tensor_runtime_bridge import triangular_solve_from_tensors

    n = LD.shape[-1]
    k = B.shape[-1]
    idx = (pivots - 1).to(dtype=torch.long)
    lead = (1,) * (B.ndim - 2)
    idx_e = idx.reshape(list(lead) + [n, 1]).expand(list(lead) + [n, k])
    Bp = torch.gather(B, -2, idx_e)
    Lmat = torch.tril(LD, -1) + torch.eye(n, dtype=LD.dtype)
    d = diagonal(LD).unsqueeze(-1)
    z = triangular_solve_from_tensors(Lmat, Bp, upper=False)
    y = z / d
    return triangular_solve_from_tensors(Lmat.transpose(-1, -2), y, upper=True)


def householder_product(A: Tensor, tau: Tensor) -> Tensor:
    """Product of Householder reflectors (composition over existing GPU matmul).

    Returns the first ``n`` columns of ``Q = H_0 H_1 ... H_{k-1}`` where
    ``H_i = I - tau_i v_i v_i^T`` and ``v_i`` is taken from column ``i`` of ``A``
    (with an implicit leading 1).
    """
    dtype = A.dtype
    *lead, m, n = A.shape
    k = tau.shape[-1]
    A_flat = A.reshape([-1, m, n])
    tau_flat = tau.reshape([-1, k])
    B = A_flat.shape[0]
    outs: list[Tensor] = []
    for b in range(B):
        Ab = A_flat.select(0, b)
        tb = tau_flat.select(0, b)
        Qb = torch.eye(m, dtype=dtype)[:, :n]
        for i in reversed(range(k)):
            v = Ab.select(1, i)[i:]
            w = torch.cat([torch.zeros(i, dtype=dtype), torch.ones(1, dtype=dtype), v[1:]])
            u = w.reshape([m, 1])
            Qb = Qb - tb[i] * u.matmul(u.transpose(0, 1)).matmul(Qb)
        outs.append(Qb)
    return torch.stack(outs, 0).reshape(list(lead) + [m, n])


def vecdot(x: Tensor, y: Tensor, *, dim: int = -1) -> Tensor:
    """Complex-safe dot product over ``dim``: ``sum(conj(x) * y, dim)``."""
    return (x.conj() * y).sum(dim)


def tensorinv(A: Tensor, ind: int = 2) -> Tensor:
    """Inverse of a tensor viewed as a stack of square matrices."""
    if A.ndim < 2:
        raise LinAlgError("tensorinv requires at least 2 dimensions")
    if ind <= 0 or ind > A.ndim:
        raise ValueError("ind must be > 0 and <= A.ndim")
    oldshape = list(A.shape)
    prod_old = 1
    for s in oldshape[:ind]:
        prod_old *= s
    prod_new = 1
    for s in oldshape[ind:]:
        prod_new *= s
    if prod_old != prod_new:
        raise LinAlgError("tensorinv requires the leading and trailing dimension products to match")
    a2 = A.reshape([prod_old, prod_new])
    ia = solve(a2, torch.eye(prod_old, dtype=A.dtype))
    return ia.reshape(oldshape[ind:] + oldshape[:ind])


def tensorsolve(A: Tensor, B: Tensor, ind: int = 2) -> Tensor:
    """Solves ``A X = B`` where ``A`` is viewed as a stack of square matrices."""
    if A.ndim < 2:
        raise LinAlgError("tensorsolve requires A with at least 2 dimensions")
    if ind <= 0 or ind > A.ndim:
        raise ValueError("ind must be > 0 and <= A.ndim")
    oldA = list(A.shape)
    oldB = list(B.shape)
    lead = 1
    for s in oldA[:ind]:
        lead *= s
    trail = 1
    for s in oldA[ind:]:
        trail *= s
    if lead != trail:
        raise LinAlgError("tensorsolve requires the leading and trailing dimension products of A to match")
    a2 = A.reshape([lead, trail])
    b2 = B.reshape([trail, -1]) if B.ndim > 0 else B.reshape([trail, 1])
    x = solve(a2, b2)
    return x.reshape(oldA[ind:] + oldB[ind:])


__all__ = [
    "svd", "qr", "eig", "eigh", "solve", "pinv", "inv",
    "det", "cholesky", "lu", "matrix_power", "matrix_rank",
    "norm", "lstsq", "cross", "slogdet", "svdvals", "diagonal",
    "eigvals", "eigvalsh", "cond", "vector_norm", "matrix_norm",
    "solve_triangular", "lu_factor", "lu_solve", "matrix_exp",
    "multi_dot", "vander", "LinAlgError", "cholesky_ex", "inv_ex",
    "lu_factor_ex", "solve_ex", "ldl_factor", "ldl_factor_ex",
    "ldl_solve", "householder_product", "vecdot", "tensorinv",
    "tensorsolve",
]
