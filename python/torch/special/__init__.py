from __future__ import annotations

import math

import torch
from torch import Tensor

_SQRT2 = math.sqrt(2.0)
_PI = math.pi


def entr(x: Tensor) -> Tensor:
    """Elementwise entropy ``-x * log(x)``; ``0`` at ``x == 0`` and ``-inf`` for ``x < 0``."""
    safe = torch.where(x > 0, x, torch.ones_like(x))
    val = -x * torch.log(safe)
    val = torch.where(x > 0, val, torch.zeros_like(x))
    return torch.where(x < 0, torch.full_like(x, float("-inf")), val)


def erf(x: Tensor) -> Tensor:
    return torch.erf(x)


def erfc(x: Tensor) -> Tensor:
    return torch.erfc(x)


def expm1(x: Tensor) -> Tensor:
    return torch.expm1(x)


def exp2(x: Tensor) -> Tensor:
    return torch.exp2(x)


def expit(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


def gammaln(x: Tensor) -> Tensor:
    return torch.lgamma(x)


def digamma(x: Tensor) -> Tensor:
    return torch.digamma(x)


psi = digamma


def i0(x: Tensor) -> Tensor:
    return torch.i0(x)


def i0e(x: Tensor) -> Tensor:
    return torch.i0(x) * torch.exp(-torch.abs(x))


def log1p(x: Tensor) -> Tensor:
    return torch.log1p(x)


def logit(x: Tensor, eps: float | None = None) -> Tensor:
    if eps is not None:
        x = torch.clamp(x, eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def xlogy(x: Tensor, y: Tensor) -> Tensor:
    return torch.xlogy(x, y)


def xlog1py(x: Tensor, y: Tensor) -> Tensor:
    safe = torch.where(x != 0, y, torch.zeros_like(y))
    return torch.where(x != 0, x * torch.log1p(safe), torch.zeros_like(x))


def ndtr(x: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def log_ndtr(x: Tensor) -> Tensor:
    return torch.log(ndtr(x))


def sinc(x: Tensor) -> Tensor:
    px = _PI * x
    safe = torch.where(x == 0, torch.ones_like(px), px)
    val = torch.sin(safe) / safe
    return torch.where(x == 0, torch.ones_like(x), val)


def round(x: Tensor) -> Tensor:
    return torch.round(x)


def softmax(x: Tensor, dim: int) -> Tensor:
    return torch.softmax(x, dim=dim)


def log_softmax(x: Tensor, dim: int) -> Tensor:
    return torch.log_softmax(x, dim=dim)


def multigammaln(x: Tensor, p: int) -> Tensor:
    acc = torch.full_like(x, 0.25 * p * (p - 1) * math.log(_PI))
    for i in range(p):
        acc = acc + torch.lgamma(x - 0.5 * i)
    return acc


__all__ = [
    "entr", "erf", "erfc", "expm1", "exp2", "expit", "gammaln",
    "digamma", "psi", "i0", "i0e", "log1p", "logit", "xlogy",
    "xlog1py", "ndtr", "log_ndtr", "sinc", "round", "softmax",
    "log_softmax", "multigammaln",
]
