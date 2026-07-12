"""Multivariate distributions for ``torch.distributions``."""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor
import torch.special as special

from .utils import (
    Distribution,
    _as_shape,
    _broadcast_all,
    _broadcast_shapes,
    _standard_normal,
)
from .constraints import real, positive_definite, lower_cholesky, simplex, Constraint
from .univariate import Normal, Categorical
from .transforms import CorrCholeskyTransform, ExpTransform, SigmoidTransform, StickBreakingTransform


class MultivariateNormal(Distribution):
    arg_constraints = {"loc": real, "covariance_matrix": positive_definite}
    support = real
    has_rsample = True
    event_dim = 1

    def __init__(
        self,
        loc: Tensor,
        covariance_matrix: Tensor | None = None,
        precision_matrix: Tensor | None = None,
        scale_tril: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if (covariance_matrix is None) + (precision_matrix is None) + (scale_tril is None) != 2:
            raise ValueError("Exactly one of covariance_matrix, precision_matrix, scale_tril")
        self.loc = loc if isinstance(loc, Tensor) else torch.tensor(loc)
        d = self.loc.shape[-1]
        if covariance_matrix is not None:
            self.covariance_matrix = covariance_matrix
            self.scale_tril = torch.linalg.cholesky(covariance_matrix)
            self._prec = torch.linalg.inv(covariance_matrix)
        elif precision_matrix is not None:
            self._prec = precision_matrix
            self.covariance_matrix = torch.linalg.inv(precision_matrix)
            self.scale_tril = torch.linalg.cholesky(self.covariance_matrix)
        else:
            self.scale_tril = scale_tril
            self.covariance_matrix = scale_tril.matmul(scale_tril.transpose(-1, -2))
            self._prec = torch.linalg.inv(self.covariance_matrix)
        self._log_det = 2.0 * self.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        super().__init__(self.loc.shape[:-1], (d,), validate_args)

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return self.covariance_matrix.diagonal(dim1=-2, dim2=-1)

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = list(_as_shape(sample_shape)) + list(self.batch_shape) + [self.event_shape[0]]
        eps = _standard_normal(shape)
        return self.loc + torch.matmul(self.scale_tril, eps.unsqueeze(-1)).squeeze(-1)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        if value.shape[-1] != self.event_shape[0]:
            raise ValueError("value event shape mismatch")
        diff = value - self.loc
        maha = torch.matmul(torch.matmul(diff.unsqueeze(-2), self._prec), diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        k = self.event_shape[0]
        return -0.5 * (k * math.log(2.0 * math.pi) + self._log_det + maha)


def _batch_capacitance_tril(W: Tensor, D: Tensor) -> Tensor:
    m = W.shape[-1]
    Wt_Dinv = W.transpose(-1, -2) / D.unsqueeze(-2)
    K = torch.matmul(Wt_Dinv, W).contiguous()
    eye = torch.eye(m, dtype=K.dtype)
    K = K + eye
    return torch.linalg.cholesky(K)


def _batch_lowrank_logdet(W: Tensor, D: Tensor, capacitance_tril: Tensor) -> Tensor:
    return 2.0 * capacitance_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1) + D.log().sum(-1)


class LowRankMultivariateNormal(Distribution):
    arg_constraints = {"loc": real, "covariance_diag": positive_definite}
    support = real
    has_rsample = True
    event_dim = 1

    def __init__(self, loc: Tensor, cov_factor: Tensor, cov_diag: Tensor, validate_args: bool | None = None) -> None:
        self.loc = loc if isinstance(loc, Tensor) else torch.tensor(loc)
        self.cov_factor = cov_factor
        self.cov_diag = cov_diag
        loc_ = self.loc.unsqueeze(-1)
        cov_diag_ = self.cov_diag.unsqueeze(-1)
        loc_, self.cov_factor, cov_diag_ = torch.broadcast_tensors(loc_, self.cov_factor, cov_diag_)
        self.loc = loc_[tuple([slice(None)] * (len(loc_.shape) - 1) + [0])]
        self.cov_diag = cov_diag_[tuple([slice(None)] * (len(cov_diag_.shape) - 1) + [0])]
        batch_shape = self.loc.shape[:-1]
        event_shape = self.loc.shape[-1:]
        self._capacitance_tril = _batch_capacitance_tril(self.cov_factor, self.cov_diag)
        super().__init__(batch_shape, event_shape, validate_args)

    @property
    def mean(self) -> Tensor:
        return self.loc

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = list(_as_shape(sample_shape)) + list(self.batch_shape) + [self.event_shape[0]]
        n = _standard_normal(shape)
        r = self.cov_factor.shape[-1]
        w_shape = list(_as_shape(sample_shape)) + list(self.batch_shape) + [r]
        w = _standard_normal(w_shape)
        return self.loc + self.cov_diag.sqrt() * n + torch.matmul(self.cov_factor, w.unsqueeze(-1)).squeeze(-1)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        if value.shape[-1] != self.event_shape[0]:
            raise ValueError("value event shape mismatch")
        diff = value - self.loc
        # Full precision matrix P = W W^T + diag(D); mathematically identical to
        # the low-rank Woodbury form (used for log_det via the matrix-determinant lemma).
        precision = torch.matmul(self.cov_factor, self.cov_factor.transpose(-1, -2)) + torch.diag_embed(self.cov_diag)
        prec_inv = torch.linalg.inv(precision)
        M = torch.matmul(torch.matmul(diff.unsqueeze(-2), prec_inv), diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        log_det = _batch_lowrank_logdet(
            self.cov_factor, self.cov_diag, self._capacitance_tril
        )
        k = self.event_shape[0]
        return -0.5 * (k * math.log(2.0 * math.pi) + log_det + M)


class LogisticNormal(Distribution):
    support = simplex
    has_rsample = True
    event_dim = 1

    def __init__(self, loc: Tensor, scale: Tensor, validate_args: bool | None = None) -> None:
        self.loc = loc if isinstance(loc, Tensor) else torch.tensor(loc)
        self.scale = scale if isinstance(scale, Tensor) else torch.tensor(scale)
        self.loc, self.scale = _broadcast_all(self.loc, self.scale)
        base_dist = Normal(self.loc, self.scale, validate_args=validate_args)
        if not base_dist.batch_shape:
            base_dist = base_dist.expand([1])
        from .distributions import TransformedDistribution

        self._td = TransformedDistribution(base_dist, StickBreakingTransform(), validate_args=validate_args)
        super().__init__(self._td.batch_shape, self._td.event_shape, validate_args)

    @property
    def mean(self) -> Tensor:
        return torch.softmax(self.loc, dim=-1)

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self._td.rsample(sample_shape)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self._td.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        return self._td.log_prob(value)


class Wishart(Distribution):
    arg_constraints = {"df": positive_definite, "covariance_matrix": positive_definite}
    support = positive_definite
    has_rsample = True
    event_dim = 2

    def __init__(self, df: Tensor | float, covariance_matrix: Tensor, validate_args: bool | None = None) -> None:
        if covariance_matrix.dim() < 2:
            raise ValueError("covariance_matrix must be at least 2D")
        self.covariance_matrix = covariance_matrix
        self.df = df if isinstance(df, Tensor) else torch.tensor(float(df))
        d = covariance_matrix.shape[-1]
        self._L = torch.linalg.cholesky(covariance_matrix)
        super().__init__(covariance_matrix.shape[:-2], (d, d), validate_args)

    @property
    def mean(self) -> Tensor:
        return self.df * self.covariance_matrix

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = list(_as_shape(sample_shape)) + list(self.batch_shape)
        d = self.event_shape[0]
        # Bartlett decomposition: Z lower-triangular with N(0,1) off-diagonal and
        # sqrt(chi2(df - i)) on the i-th diagonal (chi2(k) = 2*Gamma(k/2, 1/2)).
        Z = torch.zeros(shape + [d, d], dtype=self.covariance_matrix.dtype)
        # off-diagonal standard normals
        n_off = d * (d - 1) // 2
        if n_off > 0:
            off = _standard_normal(shape + [n_off])
            idx = 0
            for i in range(d):
                for j in range(i):
                    Z[..., i, j] = off[..., idx]
                    idx += 1
        # diagonal chi2 samples
        df_flat = (self.df - torch.arange(0.0, float(d), dtype=self.covariance_matrix.dtype))
        for i in range(d):
            gamma = Normal(torch.tensor(0.0), torch.tensor(1.0)).sample(shape)
            chi = torch.sqrt(2.0 * _gamma_sample(df_flat[..., i] / 2.0, 0.5, shape))
            Z[..., i, i] = chi
        A = self._L.matmul(Z)
        return A.matmul(A.transpose(-1, -2))

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        d = self.event_shape[0]
        df = self.df
        sign, logdet = torch.linalg.slogdet(self.covariance_matrix)
        lp = (df - d - 1.0) / 2.0 * torch.linalg.logdet(value) - df / 2.0 * logdet
        lp = lp - df * d / 2.0 * math.log(2.0) - d * (d - 1) / 4.0 * math.log(math.pi)
        lp = lp - 0.5 * torch.linalg.trace(torch.linalg.solve(self.covariance_matrix, value))
        lp = lp - _log_multigammaln(df / 2.0, d)
        return lp


def _gamma_sample(conc: Tensor, rate: Tensor, shape: list[int]) -> Tensor:
    from .univariate import Gamma
    return Gamma(conc.expand(shape), rate).sample()


def _log_multigammaln(a: Tensor, d: int) -> Tensor:
    out = torch.zeros_like(a)
    for i in range(1, d + 1):
        out = out + special.gammaln(a + (1.0 - i) / 2.0)
    return out


class LKJCholesky(Distribution):
    arg_constraints = {"correlation": lower_cholesky}
    support = lower_cholesky
    has_rsample = True
    event_dim = 2

    def __init__(self, dim: int, concentration: Tensor | float, validate_args: bool | None = None) -> None:
        self.dim = dim
        self.concentration = concentration if isinstance(concentration, Tensor) else torch.tensor(float(concentration))
        n_params = int(dim * (dim - 1) / 2)
        self._transform = CorrCholeskyTransform()
        self._n_params = n_params
        super().__init__(self.concentration.shape, (dim, dim), validate_args)

    @property
    def mean(self) -> Tensor:
        eye = torch.eye(self.dim, dtype=self.concentration.dtype)
        return eye.expand(list(self.batch_shape) + [self.dim, self.dim])

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = list(_as_shape(sample_shape)) + list(self.batch_shape) + [self._n_params]
        z = _standard_normal(shape) * torch.sqrt(2.0 * self.concentration).unsqueeze(-1)
        # CorrCholeskyTransform expects a flat real vector of length n_params.
        return self._transform(z)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        eta = self.concentration + (self.d - 1.0) / 2.0
        below = value.tril(-1)
        return (eta * torch.log1p(-below ** 2)).sum(dim=(-2, -1))


class MixtureSameFamily(Distribution):
    has_rsample = False
    event_dim = 1

    def __init__(self, mixture_distribution: Categorical, component_distribution: Distribution, validate_args: bool | None = None) -> None:
        self.mixture_distribution = mixture_distribution
        self.component_distribution = component_distribution
        self._event_shape = list(component_distribution.event_shape)
        comp_batch = list(component_distribution.batch_shape[:-1]) if component_distribution.batch_shape else []
        self._batch_shape = _broadcast_shapes(list(mixture_distribution.batch_shape), comp_batch)
        super().__init__(self._batch_shape, self._event_shape, validate_args)

    @property
    def mean(self) -> Tensor:
        probs = self.mixture_distribution.probs
        comp = self.component_distribution.mean
        return (probs.unsqueeze(-1) * comp).sum(dim=-2)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        with torch.no_grad():
            mix = self.mixture_distribution.sample(sample_shape)
            comp = self.component_distribution.sample(sample_shape)
            idx = mix.reshape(list(mix.shape) + [1, 1])
            return comp.gather(-2, idx.expand(list(comp.shape[:-2]) + [1, self._event_shape[0]]).long()).squeeze(-2)

    def log_prob(self, value: Tensor) -> Tensor:
        event_ndims = len(self._event_shape)
        value = value.unsqueeze(-1 - event_ndims)
        comp_lp = self.component_distribution.log_prob(value)
        mix_lp = torch.log_softmax(self.mixture_distribution.logits, dim=-1)
        return torch.logsumexp(comp_lp + mix_lp, dim=-1)


__all__ = [
    "MultivariateNormal",
    "LowRankMultivariateNormal",
    "LogisticNormal",
    "Wishart",
    "LKJCholesky",
    "MixtureSameFamily",
]
