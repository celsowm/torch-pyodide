"""Discrete Fourier transforms (``torch.fft``).

Implemented on top of the real-valued GPU ops by materializing DFT
cosine/sine matrices and evaluating the transform as matrix products.
Complex inputs/outputs use the ``complex64`` representation from
``torch._complex`` (interleaved real/imag float32 storage).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from .._complex import (
    is_complex,
    real as _real,
    imag as _imag,
    complex as _complex,
    conj as _conj,
    _real_view,
    _complex_from_real_view,
)


def _prod(values: list[int]) -> int:
    out = 1
    for v in values:
        out *= v
    return out


def _dft_matrices(n: int) -> tuple[Tensor, Tensor]:
    idx = torch.arange(n, dtype="float32")
    grid = idx.reshape([n, 1]) * idx.reshape([1, n])
    ang = grid * (2.0 * math.pi / n)
    return torch.cos(ang), torch.sin(ang)


def _scale(n: int, inverse: bool, norm: str | None) -> float:
    if norm is None or norm == "backward":
        return 1.0 / n if inverse else 1.0
    if norm == "forward":
        return 1.0 if inverse else 1.0 / n
    if norm == "ortho":
        return 1.0 / math.sqrt(n)
    raise ValueError(f"Invalid norm value {norm!r}")


def _fix_len(x: Tensor, n: int) -> Tensor:
    cur = x.shape[-1]
    if n == cur:
        return x
    if n < cur:
        return x.slice(-1, 0, n)
    pad = torch.zeros(list(x.shape[:-1]) + [n - cur], dtype="float32")
    return torch.cat([x, pad], dim=-1)


def _fft_1d(x: Tensor, n: int | None, dim: int, inverse: bool, norm: str | None) -> Tensor:
    ndim = x.ndim
    dim = dim % ndim
    if is_complex(x):
        xr0, xi0 = _real(x), _imag(x)
    else:
        xr0, xi0 = x, torch.zeros_like(x)

    last = ndim - 1
    if dim != last:
        xr0 = xr0.transpose(dim, last)
        xi0 = xi0.transpose(dim, last)
    t_shape = list(xr0.shape)
    cur = t_shape[-1]
    N = cur if n is None else n

    P = _prod(t_shape[:-1]) if len(t_shape) > 1 else 1
    xr = xr0.reshape([P, cur])
    xi = xi0.reshape([P, cur])
    if N != cur:
        xr = _fix_len(xr, N)
        xi = _fix_len(xi, N)

    Wc, Ws = _dft_matrices(N)
    if not inverse:
        Xr = xr.matmul(Wc) + xi.matmul(Ws)
        Xi = xi.matmul(Wc) - xr.matmul(Ws)
    else:
        Xr = xr.matmul(Wc) - xi.matmul(Ws)
        Xi = xi.matmul(Wc) + xr.matmul(Ws)

    s = _scale(N, inverse, norm)
    if s != 1.0:
        Xr = Xr * s
        Xi = Xi * s

    out_shape = t_shape[:-1] + [N]
    Xr = Xr.reshape(out_shape)
    Xi = Xi.reshape(out_shape)
    if dim != last:
        Xr = Xr.transpose(dim, last)
        Xi = Xi.transpose(dim, last)
    return _complex(Xr, Xi)


# ── complex shape helpers (operate on the shared real view) ──────────

def _c_slice(z: Tensor, dim: int, start: int, end: int) -> Tensor:
    d = dim % z.ndim
    return _complex_from_real_view(_real_view(z).slice(d, start, end))


def _c_flip(z: Tensor, dim: int) -> Tensor:
    d = dim % z.ndim
    return _complex_from_real_view(_real_view(z).flip([d]))


def _c_cat(parts: list[Tensor], dim: int) -> Tensor:
    d = dim % parts[0].ndim
    return _complex_from_real_view(torch.cat([_real_view(p) for p in parts], dim=d))


# ── public 1-D transforms ────────────────────────────────────────────

def fft(input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None) -> Tensor:
    return _fft_1d(input, n, dim, inverse=False, norm=norm)


def ifft(input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None) -> Tensor:
    return _fft_1d(input, n, dim, inverse=True, norm=norm)


def rfft(input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None) -> Tensor:
    d = dim % input.ndim
    N = input.shape[d] if n is None else n
    full = _fft_1d(input, N, d, inverse=False, norm=norm)
    keep = N // 2 + 1
    return _c_slice(full, d, 0, keep)


def irfft(input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None) -> Tensor:
    d = dim % input.ndim
    m = input.shape[d]
    N = 2 * (m - 1) if n is None else n
    half = input if is_complex(input) else _complex(input, torch.zeros_like(input))
    if N <= m:
        full = _c_slice(half, d, 0, N)
    else:
        tail_src = _c_slice(half, d, 1, N - m + 1)
        tail = _conj(_c_flip(tail_src, d))
        full = _c_cat([half, tail], d)
    out = _fft_1d(full, N, d, inverse=True, norm=norm)
    return _real(out)


# ── n-dimensional transforms ─────────────────────────────────────────

def _resolve_dims(input: Tensor, dim) -> list[int]:
    if dim is None:
        return list(range(input.ndim))
    if isinstance(dim, int):
        return [dim % input.ndim]
    return [d % input.ndim for d in dim]


def fftn(input: Tensor, s=None, dim=None, norm: str | None = None) -> Tensor:
    dims = _resolve_dims(input, dim)
    sizes = [None] * len(dims) if s is None else list(s)
    out = input
    for k, d in enumerate(dims):
        out = _fft_1d(out, sizes[k], d, inverse=False, norm=norm)
    return out


def ifftn(input: Tensor, s=None, dim=None, norm: str | None = None) -> Tensor:
    dims = _resolve_dims(input, dim)
    sizes = [None] * len(dims) if s is None else list(s)
    out = input
    for k, d in enumerate(dims):
        out = _fft_1d(out, sizes[k], d, inverse=True, norm=norm)
    return out


def fft2(input: Tensor, s=None, dim=(-2, -1), norm: str | None = None) -> Tensor:
    return fftn(input, s=s, dim=dim, norm=norm)


def ifft2(input: Tensor, s=None, dim=(-2, -1), norm: str | None = None) -> Tensor:
    return ifftn(input, s=s, dim=dim, norm=norm)


def rfftn(input: Tensor, s=None, dim=None, norm: str | None = None) -> Tensor:
    dims = _resolve_dims(input, dim)
    sizes = [None] * len(dims) if s is None else list(s)
    out = rfft(input, sizes[-1], dims[-1], norm=norm)
    for k in range(len(dims) - 2, -1, -1):
        out = _fft_1d(out, sizes[k], dims[k], inverse=False, norm=norm)
    return out


def rfft2(input: Tensor, s=None, dim=(-2, -1), norm: str | None = None) -> Tensor:
    return rfftn(input, s=s, dim=dim, norm=norm)


def irfftn(input: Tensor, s=None, dim=None, norm: str | None = None) -> Tensor:
    dims = _resolve_dims(input, dim)
    sizes = [None] * len(dims) if s is None else list(s)
    out = input if is_complex(input) else _complex(input, torch.zeros_like(input))
    for k in range(len(dims) - 1):
        out = _fft_1d(out, sizes[k], dims[k], inverse=True, norm=norm)
    return irfft(out, sizes[-1], dims[-1], norm=norm)


def irfft2(input: Tensor, s=None, dim=(-2, -1), norm: str | None = None) -> Tensor:
    return irfftn(input, s=s, dim=dim, norm=norm)


# ── Hermitian-symmetric transforms ───────────────────────────────────
# hfft/ihfft mirror irfft/rfft with the forward/backward normalization
# swapped (ortho is self-inverse). Verified against torch.fft reference.

def _swap_norm(norm: str | None) -> str:
    if norm is None or norm == "backward":
        return "forward"
    if norm == "forward":
        return "backward"
    return norm


def hfft(input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None) -> Tensor:
    d = dim % input.ndim
    z = input if is_complex(input) else _complex(input, torch.zeros_like(input))
    return irfft(_conj(z), n=n, dim=d, norm=_swap_norm(norm))


def ihfft(input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None) -> Tensor:
    d = dim % input.ndim
    return _conj(rfft(input, n=n, dim=d, norm=_swap_norm(norm)))


def hfftn(input: Tensor, s=None, dim=None, norm: str | None = None) -> Tensor:
    z = input if is_complex(input) else _complex(input, torch.zeros_like(input))
    return irfftn(_conj(z), s=s, dim=dim, norm=_swap_norm(norm))


def hfft2(input: Tensor, s=None, dim=(-2, -1), norm: str | None = None) -> Tensor:
    return hfftn(input, s=s, dim=dim, norm=norm)


def ihfftn(input: Tensor, s=None, dim=None, norm: str | None = None) -> Tensor:
    return _conj(rfftn(input, s=s, dim=dim, norm=_swap_norm(norm)))


def ihfft2(input: Tensor, s=None, dim=(-2, -1), norm: str | None = None) -> Tensor:
    return ihfftn(input, s=s, dim=dim, norm=norm)


# ── helpers / frequency utilities ────────────────────────────────────

def fftfreq(n: int, d: float = 1.0, dtype: str = "float32") -> Tensor:
    vals = []
    cutoff = (n - 1) // 2 + 1
    for i in range(cutoff):
        vals.append(i)
    for i in range(-(n // 2), 0):
        vals.append(i)
    scale = 1.0 / (n * d)
    return torch.tensor([v * scale for v in vals], dtype=dtype)


def rfftfreq(n: int, d: float = 1.0, dtype: str = "float32") -> Tensor:
    scale = 1.0 / (n * d)
    return torch.tensor([i * scale for i in range(n // 2 + 1)], dtype=dtype)


def _roll_1d(z: Tensor, shift: int, dim: int) -> Tensor:
    complex_input = is_complex(z)
    n = z.shape[dim % z.ndim]
    if n == 0:
        return z
    shift = shift % n
    if shift == 0:
        return z
    d = dim % z.ndim
    if complex_input:
        a = _c_slice(z, d, n - shift, n)
        b = _c_slice(z, d, 0, n - shift)
        return _c_cat([a, b], d)
    a = z.slice(d, n - shift, n)
    b = z.slice(d, 0, n - shift)
    return torch.cat([a, b], dim=d)


def fftshift(input: Tensor, dim=None) -> Tensor:
    dims = _resolve_dims(input, dim)
    out = input
    for d in dims:
        out = _roll_1d(out, input.shape[d] // 2, d)
    return out


def ifftshift(input: Tensor, dim=None) -> Tensor:
    dims = _resolve_dims(input, dim)
    out = input
    for d in dims:
        n = input.shape[d]
        out = _roll_1d(out, -(n // 2), d)
    return out


__all__ = [
    "fft", "ifft", "rfft", "irfft",
    "fft2", "ifft2", "fftn", "ifftn", "rfftn", "rfft2",
    "irfftn", "irfft2",
    "hfft", "ihfft", "hfftn", "hfft2", "ihfftn", "ihfft2",
    "fftfreq", "rfftfreq", "fftshift", "ifftshift",
]
