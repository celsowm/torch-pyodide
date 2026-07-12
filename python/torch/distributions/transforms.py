"""Transforms for ``torch.distributions``.

Each transform implements ``forward``, ``inv`` (inverse) and
``log_abs_det_jacobian``. ``Transform`` carries ``domain`` / ``codomain``
constraints and ``event_dim`` so ``TransformedDistribution`` can compose them.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .constraints import (
    Constraint,
    real,
    positive,
    unit_interval,
    simplex,
    real_vector,
    lower_cholesky,
    positive_definite,
)


class Transform:
    domain: Constraint = real
    codomain: Constraint = real
    event_dim: int = 0
    bijective: bool = True

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def inv(self, y: Tensor) -> Tensor:
        raise NotImplementedError

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError

    def forward_shape(self, shape: list) -> list:
        return shape

    def inverse_shape(self, shape: list) -> list:
        return shape


class IdentityTransform(Transform):
    def forward(self, x: Tensor) -> Tensor:
        return x

    def inv(self, y: Tensor) -> Tensor:
        return y

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.zeros_like(x)


class ExpTransform(Transform):
    domain = real
    codomain = positive

    def forward(self, x: Tensor) -> Tensor:
        return x.exp()

    def inv(self, y: Tensor) -> Tensor:
        return y.log()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return y.log()


class LogTransform(Transform):
    domain = positive
    codomain = real

    def forward(self, x: Tensor) -> Tensor:
        return x.log()

    def inv(self, y: Tensor) -> Tensor:
        return y.exp()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return -y


class PowerTransform(Transform):
    domain = positive
    codomain = positive

    def __init__(self, exponent: float) -> None:
        self.exponent = exponent

    def forward(self, x: Tensor) -> Tensor:
        return x.pow(self.exponent)

    def inv(self, y: Tensor) -> Tensor:
        return y.pow(1.0 / self.exponent)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.abs(torch.log(self.exponent) + (self.exponent - 1.0) * x.log())


class SigmoidTransform(Transform):
    domain = real
    codomain = unit_interval

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

    def inv(self, y: Tensor) -> Tensor:
        return y.log() - (-y).log1p()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return -torch.nn.functional.softplus(-x) - torch.nn.functional.softplus(x)


class TanhTransform(Transform):
    domain = real
    codomain = real

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

    def inv(self, y: Tensor) -> Tensor:
        return torch.atanh(y)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return 2.0 * (math.log(2.0) - x - torch.nn.functional.softplus(-2.0 * x))


class SoftmaxTransform(Transform):
    domain = real
    codomain = simplex
    event_dim = 1

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(dim=-1)

    def inv(self, y: Tensor) -> Tensor:
        return y.log()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return (-y * y.log()).sum(dim=-1)


class SoftplusTransform(Transform):
    domain = real
    codomain = positive

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.softplus(x)

    def inv(self, y: Tensor) -> Tensor:
        return y + torch.log(-torch.expm1(-y))

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return -torch.nn.functional.softplus(-x)


class AbsTransform(Transform):
    domain = real
    codomain = positive

    def forward(self, x: Tensor) -> Tensor:
        return x.abs()

    def inv(self, y: Tensor) -> Tensor:
        return y

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.zeros_like(x)


class AffineTransform(Transform):
    def __init__(self, loc: Tensor | float, scale: Tensor | float, event_dim: int = 0) -> None:
        self.loc = loc if isinstance(loc, Tensor) else torch.tensor(float(loc))
        self.scale = scale if isinstance(scale, Tensor) else torch.tensor(float(scale))
        self.event_dim = event_dim

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale + self.loc

    def inv(self, y: Tensor) -> Tensor:
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return self.scale.abs().log()


class ComposeTransform(Transform):
    def __init__(self, transforms: list[Transform]) -> None:
        self.parts = list(transforms)
        self.event_dim = max((t.event_dim for t in self.parts), default=0)

    @property
    def domain(self) -> Constraint:
        return self.parts[0].domain if self.parts else real

    @property
    def codomain(self) -> Constraint:
        return self.parts[-1].codomain if self.parts else real

    def forward(self, x: Tensor) -> Tensor:
        for t in self.parts:
            x = t(x)
        return x

    def inv(self, y: Tensor) -> Tensor:
        for t in reversed(self.parts):
            y = t.inv(y)
        return y

    def forward_shape(self, shape: list) -> list:
        for part in self.parts:
            shape = part.forward_shape(shape)
        return shape

    def inverse_shape(self, shape: list) -> list:
        for part in reversed(self.parts):
            shape = part.inverse_shape(shape)
        return shape

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        # Approximate: sum per-part LADJ along the chain using recomputed points.
        total = torch.zeros_like(x)
        cur = x
        for t in self.parts:
            nxt = t(cur)
            total = total + t.log_abs_det_jacobian(cur, nxt)
            cur = nxt
        return total


class CatTransform(Transform):
    def __init__(self, transforms: list[Transform], dim: int = -1) -> None:
        self.parts = list(transforms)
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([t(x) for t in self.parts], dim=self.dim)

    def inv(self, y: Tensor) -> Tensor:
        return torch.cat([t.inv(y) for t in self.parts], dim=self.dim)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.zeros_like(x)


class StackTransform(Transform):
    def __init__(self, transforms: list[Transform], dim: int = 0) -> None:
        self.parts = list(transforms)
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([t(x) for t in self.parts], dim=self.dim)

    def inv(self, y: Tensor) -> Tensor:
        return torch.stack([t.inv(y.select(self.dim, i)) for i, t in enumerate(self.parts)], dim=self.dim)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.zeros_like(x)


class ReshapeTransform(Transform):
    def __init__(self, in_shape: list[int], out_shape: list[int]) -> None:
        self.in_shape = list(in_shape)
        self.out_shape = list(out_shape)
        self.event_dim = len(self.in_shape)

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(self.out_shape)

    def inv(self, y: Tensor) -> Tensor:
        return y.reshape(self.in_shape)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.zeros_like(x)


class IndependentTransform(Transform):
    def __init__(self, base_transform: Transform, reinterpreted_batch_ndims: int) -> None:
        self.base_transform = base_transform
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        self.event_dim = base_transform.event_dim + reinterpreted_batch_ndims

    def forward(self, x: Tensor) -> Tensor:
        return self.base_transform(x)

    def inv(self, y: Tensor) -> Tensor:
        return self.base_transform.inv(y)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return self.base_transform.log_abs_det_jacobian(x, y)


class StickBreakingTransform(Transform):
    domain = real_vector
    codomain = simplex
    event_dim = 1

    def forward(self, x: Tensor) -> Tensor:
        # x has last dim K-1; produce a simplex of size K (matches PyTorch).
        x = torch.clamp(x, -12.0, 12.0)
        offset = x.shape[-1] + 1 - torch.ones(x.shape[-1]).cumsum(-1)
        z = torch.sigmoid(x - offset.log())
        z_cumprod = (1.0 - z).cumprod(-1)
        y = torch.nn.functional.pad(z, [0, 1], value=1) * torch.nn.functional.pad(z_cumprod, [1, 0], value=1)
        return y

    def inv(self, y: Tensor) -> Tensor:
        y_crop = y[tuple([slice(None)] * (len(y.shape) - 1) + [slice(None, -1)])]
        K = y.shape[-1]
        offset = K - torch.ones(y_crop.shape[-1]).cumsum(-1)
        sf = 1.0 - y_crop.cumsum(-1)
        sf = torch.clamp(sf, min=torch.finfo(y.dtype).tiny)
        return y_crop.log() - sf.log() + offset.log()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        offset = x.shape[-1] + 1 - torch.ones(x.shape[-1]).cumsum(-1)
        x = x - offset.log()
        y_crop = y[tuple([slice(None)] * (len(y.shape) - 1) + [slice(None, -1)])]
        return (-x + torch.nn.functional.logsigmoid(x) + y_crop.log()).sum(-1)

    def forward_shape(self, shape: list) -> list:
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape[:-1] + [shape[-1] + 1]

    def inverse_shape(self, shape: list) -> list:
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape[:-1] + [shape[-1] - 1]


class LowerCholeskyTransform(Transform):
    domain = real
    codomain = lower_cholesky
    event_dim = 2

    def forward(self, x: Tensor) -> Tensor:
        # x is a square matrix in the last two dims; produce lower-triangular
        # with positive diagonal via exp on the diagonal.
        L = x.tril()
        diag = torch.exp(x.diagonal(dim1=-2, dim2=-1))
        L = L + torch.diag_embed(diag - L.diagonal(dim1=-2, dim2=-1))
        return L

    def inv(self, y: Tensor) -> Tensor:
        return y.tril()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        diag = y.diagonal(dim1=-2, dim2=-1)
        return diag.log().sum(dim=-1)


class PositiveDefiniteTransform(Transform):
    domain = lower_cholesky
    codomain = positive_definite
    event_dim = 2

    def forward(self, x: Tensor) -> Tensor:
        return x.matmul(x.transpose(-1, -2))

    def inv(self, y: Tensor) -> Tensor:
        return torch.linalg.cholesky(y)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        diag = x.diagonal(dim1=-2, dim2=-1)
        return 2.0 * diag.log().sum(dim=-1)


class CumulativeDistributionTransform(Transform):
    domain = real
    codomain = unit_interval
    event_dim = 1

    def __init__(self, cdf: object) -> None:
        self.cdf = cdf

    def forward(self, x: Tensor) -> Tensor:
        return self.cdf(x)

    def inv(self, y: Tensor) -> Tensor:
        raise NotImplementedError("inverse CDF transform is not analytically invertible")

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError


class CorrCholeskyTransform(Transform):
    domain = real
    codomain = lower_cholesky
    event_dim = 2

    def forward(self, x: Tensor) -> Tensor:
        # x has last dim (D-1)*D/2; produce a DxD correlation Cholesky.
        D = int((1 + math.sqrt(1 + 8 * x.shape[-1])) / 2)
        L = torch.zeros(x.shape[:-1] + [D, D])
        idx = 0
        for i in range(D):
            for j in range(i + 1):
                if i == j:
                    L[..., i, i] = 1.0
                else:
                    z = x[..., idx]
                    idx += 1
                    # partial product of sqrt(1 - z_k^2) along the row
                    factor = 1.0
                    for k in range(j):
                        factor = factor * torch.sqrt(1.0 - L[..., i, k] ** 2)
                    L[..., i, j] = torch.sin(z) * factor
        return L

    def inv(self, y: Tensor) -> Tensor:
        raise NotImplementedError

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        # |J| = prod_i prod_{j<i} cos(z_ij) = prod of sqrt(1 - y_ij^2)
        below = y.tril(-1)
        return torch.log(1.0 - below ** 2).sum(dim=(-2, -1)) / 2.0


# Build the constraint proxy needed by TransformedDistribution
_constraint_proxy = real


def _transform_constraint(transforms) -> Constraint:
    return _constraint_proxy


__all__ = [
    "Transform",
    "IdentityTransform",
    "ExpTransform",
    "LogTransform",
    "PowerTransform",
    "SigmoidTransform",
    "TanhTransform",
    "SoftmaxTransform",
    "SoftplusTransform",
    "AbsTransform",
    "AffineTransform",
    "ComposeTransform",
    "CatTransform",
    "StackTransform",
    "ReshapeTransform",
    "IndependentTransform",
    "StickBreakingTransform",
    "LowerCholeskyTransform",
    "PositiveDefiniteTransform",
    "CumulativeDistributionTransform",
    "CorrCholeskyTransform",
]
