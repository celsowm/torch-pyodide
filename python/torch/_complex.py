"""Complex-number support (complex64 / complex128).

A complex tensor is represented purely at the Python level as a
reinterpretation of an underlying ``float32`` tensor whose last dimension
has size 2 (interleaved real / imaginary parts).  A complex tensor's
``_id`` therefore points at a ``float32`` runtime tensor of shape
``logical_shape + [2]`` while its own ``_shape`` is ``logical_shape`` and
its ``_dtype`` is ``"complex64"`` / ``"complex128"``.

All complex arithmetic is composed from real-valued GPU ops, so no
complex-aware shaders or runtime changes are required.
"""

from __future__ import annotations

import builtins

from ._tensor import Tensor

COMPLEX_DTYPES = ("complex64", "complex128")
builtins_complex = builtins.complex


def is_complex(t: object) -> bool:
    return isinstance(t, Tensor) and t._dtype in COMPLEX_DTYPES


def _real_view(z: Tensor) -> Tensor:
    """View a complex tensor as float32 ``[..., 2]`` sharing storage."""
    return Tensor(z._id, list(z._shape) + [2], "float32")


def _complex_from_real_view(x: Tensor, dtype: str = "complex64") -> Tensor:
    """View a float32 ``[..., 2]`` tensor as a complex tensor ``[...]``."""
    if not x._shape or x._shape[-1] != 2:
        raise RuntimeError(
            "Tensor must have a last dimension of size 2 to view as complex"
        )
    return Tensor(x._id, list(x._shape[:-1]), dtype)


def view_as_real(z: Tensor) -> Tensor:
    if not is_complex(z):
        raise RuntimeError("view_as_real is only supported for complex tensors")
    return _real_view(z)


def view_as_complex(x: Tensor) -> Tensor:
    if is_complex(x):
        return x
    return _complex_from_real_view(x)


def complex(real: Tensor, imag: Tensor, dtype: str = "complex64") -> Tensor:
    import torch

    stacked = torch.stack([real, imag], dim=-1)
    return _complex_from_real_view(stacked, dtype)


def polar(abs_: Tensor, angle_: Tensor, dtype: str = "complex64") -> Tensor:
    import torch

    return complex(abs_ * torch.cos(angle_), abs_ * torch.sin(angle_), dtype)


def real(z: Tensor) -> Tensor:
    if is_complex(z):
        return _real_view(z).select(-1, 0)
    return z


def imag(z: Tensor) -> Tensor:
    import torch

    if is_complex(z):
        return _real_view(z).select(-1, 1)
    return torch.zeros_like(z)


def conj(z: Tensor) -> Tensor:
    if not is_complex(z):
        return z
    return complex(real(z), imag(z).neg(), z._dtype)


def _promote(t: object, dtype: str = "complex64") -> Tensor:
    import torch

    if is_complex(t):
        return t
    if isinstance(t, Tensor):
        return complex(t, torch.zeros_like(t), dtype)
    # Python scalar (real or complex)
    py = builtins_complex(t)
    re = torch.tensor(float(py.real))
    im = torch.tensor(float(py.imag))
    return complex(re, im, dtype)


def _result_dtype(a: object, b: object) -> str:
    for x in (a, b):
        if is_complex(x) and x._dtype == "complex128":
            return "complex128"
    return "complex64"


def add(a: object, b: object) -> Tensor:
    dt = _result_dtype(a, b)
    ca, cb = _promote(a, dt), _promote(b, dt)
    return _complex_from_real_view(_real_view(ca) + _real_view(cb), dt)


def sub(a: object, b: object) -> Tensor:
    dt = _result_dtype(a, b)
    ca, cb = _promote(a, dt), _promote(b, dt)
    return _complex_from_real_view(_real_view(ca) - _real_view(cb), dt)


def neg(z: Tensor) -> Tensor:
    return _complex_from_real_view(_real_view(z).neg(), z._dtype)


def mul(a: object, b: object) -> Tensor:
    dt = _result_dtype(a, b)
    ca, cb = _promote(a, dt), _promote(b, dt)
    ar, ai = real(ca), imag(ca)
    br, bi = real(cb), imag(cb)
    return complex(ar * br - ai * bi, ar * bi + ai * br, dt)


def div(a: object, b: object) -> Tensor:
    dt = _result_dtype(a, b)
    ca, cb = _promote(a, dt), _promote(b, dt)
    ar, ai = real(ca), imag(ca)
    br, bi = real(cb), imag(cb)
    denom = br * br + bi * bi
    return complex((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom, dt)


def abs(z: Tensor) -> Tensor:
    r, i = real(z), imag(z)
    return (r * r + i * i).sqrt()


def angle(z: Tensor) -> Tensor:
    import torch

    if not is_complex(z):
        return torch.atan2(torch.zeros_like(z), z)
    return torch.atan2(imag(z), real(z))


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Complex matrix multiply composed from four real matmuls."""
    dt = _result_dtype(a, b)
    ca, cb = _promote(a, dt), _promote(b, dt)
    ar, ai = real(ca), imag(ca)
    br, bi = real(cb), imag(cb)
    rr = ar.matmul(br) - ai.matmul(bi)
    ii = ar.matmul(bi) + ai.matmul(br)
    return complex(rr, ii, dt)


def to_complex_list(z: Tensor) -> object:
    """Reconstruct nested Python ``complex`` values from a complex tensor."""
    rv = _real_view(z)
    nested = rv.tolist()
    depth = len(z._shape)

    def build(node: object, d: int) -> object:
        if d == 0:
            return builtins_complex(float(node[0]), float(node[1]))  # type: ignore[index]
        return [build(child, d - 1) for child in node]  # type: ignore[union-attr]

    return build(nested, depth)
