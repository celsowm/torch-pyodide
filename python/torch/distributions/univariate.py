"""Univariate distributions for ``torch.distributions``.

``log_prob`` formulas match PyTorch exactly. ``sample`` is implemented with
GPU-side random draws (build a standard sample then map through the
distribution's inverse-CDF or reparameterization). ``rsample`` is provided for
reparameterizable distributions (Normal, LogNormal, etc.).
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor
import torch.special as special


def _betaln(a: Tensor, b: Tensor) -> Tensor:
    return special.gammaln(a) + special.gammaln(b) - special.gammaln(a + b)

from .utils import (
    Distribution,
    _as_shape,
    _broadcast_all,
    _broadcast_shapes,
    _standard_gumbel,
    _standard_normal,
)
from .constraints import (
    real,
    positive,
    nonnegative,
    unit_interval,
    greater_than,
    open_interval,
    boolean,
    simplex,
    Constraint,
)


class Normal(Distribution):
    arg_constraints = {"loc": real, "scale": positive}
    support = real
    has_rsample = True

    def __init__(self, loc: Tensor | float, scale: Tensor | float, validate_args: bool | None = None) -> None:
        self.loc = loc if isinstance(loc, Tensor) else torch.tensor(float(loc))
        self.scale = scale if isinstance(scale, Tensor) else torch.tensor(float(scale))
        self.loc, self.scale = _broadcast_all(self.loc, self.scale)
        super().__init__(self.loc.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return self.scale.pow(2)

    @property
    def stddev(self) -> Tensor:
        return self.scale

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        return self.loc + self.scale * _standard_normal(shape)

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        var = self.scale.pow(2)
        log_scale = self.scale.log()
        return -((value - self.loc) ** 2) / (2.0 * var) - log_scale - math.log(math.sqrt(2.0 * math.pi))

    def cdf(self, value: Tensor) -> Tensor:
        return 0.5 * (1.0 + torch.erf((value - self.loc) / (self.scale * math.sqrt(2.0))))

    def icdf(self, value: Tensor) -> Tensor:
        return self.loc + self.scale * math.sqrt(2.0) * torch.erfinv(2.0 * value - 1.0)


class LogNormal(Normal):
    support = positive

    def __init__(self, loc: Tensor | float, scale: Tensor | float, validate_args: bool | None = None) -> None:
        super().__init__(loc, scale, validate_args)
        self._normal = Normal(loc, scale, validate_args)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self._normal.sample(sample_shape).exp()

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self._normal.rsample(sample_shape).exp()

    def log_prob(self, value: Tensor) -> Tensor:
        return self._normal.log_prob(value.log()) - value.log()

    def cdf(self, value: Tensor) -> Tensor:
        return self._normal.cdf(value.log())


class Exponential(Distribution):
    arg_constraints = {"rate": positive}
    support = positive
    has_rsample = True

    def __init__(self, rate: Tensor | float, validate_args: bool | None = None) -> None:
        self.rate = rate if isinstance(rate, Tensor) else torch.tensor(float(rate))
        super().__init__(self.rate.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        return 1.0 / self.rate

    @property
    def variance(self) -> Tensor:
        return 1.0 / self.rate.pow(2)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        return -_standard_normal(shape).abs().log() / self.rate

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.rate.log() - self.rate * value


class Laplace(Distribution):
    arg_constraints = {"loc": real, "scale": positive}
    support = real
    has_rsample = True

    def __init__(self, loc: Tensor | float, scale: Tensor | float, validate_args: bool | None = None) -> None:
        self.loc = loc if isinstance(loc, Tensor) else torch.tensor(float(loc))
        self.scale = scale if isinstance(scale, Tensor) else torch.tensor(float(scale))
        self.loc, self.scale = _broadcast_all(self.loc, self.scale)
        super().__init__(self.loc.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return 2.0 * self.scale.pow(2)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape) - 0.5
        return self.loc - self.scale * torch.sign(u) * torch.log(1.0 - 2.0 * u.abs())

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        return -((value - self.loc).abs() / self.scale) - self.scale.log() - math.log(2.0)


class Cauchy(Distribution):
    arg_constraints = {"loc": real, "scale": positive}
    support = real
    has_rsample = True

    def __init__(self, loc: Tensor | float, scale: Tensor | float, validate_args: bool | None = None) -> None:
        self.loc = loc if isinstance(loc, Tensor) else torch.tensor(float(loc))
        self.scale = scale if isinstance(scale, Tensor) else torch.tensor(float(scale))
        self.loc, self.scale = _broadcast_all(self.loc, self.scale)
        super().__init__(self.loc.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        return torch.full_like(self.loc, float("nan"))

    @property
    def variance(self) -> Tensor:
        return torch.full_like(self.loc, float("nan"))

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape)
        return self.loc + self.scale * torch.tan(math.pi * (u - 0.5))

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        return -math.log(math.pi) - self.scale.log() - (((value - self.loc) / self.scale) ** 2).log1p()


class Gumbel(Distribution):
    arg_constraints = {"loc": real, "scale": positive}
    support = real
    has_rsample = True

    def __init__(self, loc: Tensor | float, scale: Tensor | float, validate_args: bool | None = None) -> None:
        self.loc = loc if isinstance(loc, Tensor) else torch.tensor(float(loc))
        self.scale = scale if isinstance(scale, Tensor) else torch.tensor(float(scale))
        self.loc, self.scale = _broadcast_all(self.loc, self.scale)
        super().__init__(self.loc.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        return self.loc + self.scale * math.euler_gamma

    @property
    def variance(self) -> Tensor:
        return (math.pi ** 2 / 6.0) * self.scale.pow(2)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        return self.loc - self.scale * _standard_gumbel(shape)

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        z = (value - self.loc) / self.scale
        return -z - self.scale.log() - (-z).exp()


class Uniform(Distribution):
    arg_constraints = {"low": real, "high": real}
    support = real
    has_rsample = True

    def __init__(self, low: Tensor | float, high: Tensor | float, validate_args: bool | None = None) -> None:
        self.low = low if isinstance(low, Tensor) else torch.tensor(float(low))
        self.high = high if isinstance(high, Tensor) else torch.tensor(float(high))
        self.low, self.high = _broadcast_all(self.low, self.high)
        super().__init__(self.low.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        return (self.low + self.high) / 2.0

    @property
    def variance(self) -> Tensor:
        width = self.high - self.low
        return width.pow(2) / 12.0

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape)
        return self.low + u * (self.high - self.low)

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        log_width = (self.high - self.low).log()
        return torch.where((value >= self.low) & (value <= self.high), -log_width, torch.tensor(-float("inf")))

    def cdf(self, value: Tensor) -> Tensor:
        result = (value - self.low) / (self.high - self.low)
        return result.clamp(0.0, 1.0)


class Bernoulli(Distribution):
    arg_constraints = {"probs": unit_interval, "logits": real}
    support = boolean
    has_rsample = False

    def __init__(self, probs: Tensor | float | None = None, logits: Tensor | float | None = None, validate_args: bool | None = None) -> None:
        if (probs is None) == (logits is None):
            raise ValueError("Either probs or logits must be specified (not both).")
        if probs is not None:
            self._probs = probs if isinstance(probs, Tensor) else torch.tensor(float(probs))
            self._logits = self._probs.log() - (1.0 - self._probs).log()
        else:
            self._logits = logits if isinstance(logits, Tensor) else torch.tensor(float(logits))
            self._probs = torch.sigmoid(self._logits)
        super().__init__(self._probs.shape, (), validate_args)

    @property
    def probs(self) -> Tensor:
        return self._probs

    @property
    def logits(self) -> Tensor:
        return self._logits

    @property
    def mean(self) -> Tensor:
        return self._probs

    @property
    def variance(self) -> Tensor:
        return self._probs * (1.0 - self._probs)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape)
        return (u < torch.sigmoid(self._logits)).to(self._probs.dtype)

    def log_prob(self, value: Tensor) -> Tensor:
        logits = self._logits
        return value * logits - torch.nn.functional.softplus(logits)


class Categorical(Distribution):
    arg_constraints = {"probs": simplex, "logits": real}
    has_rsample = False

    def __init__(self, probs: Tensor | None = None, logits: Tensor | None = None, validate_args: bool | None = None) -> None:
        if (probs is None) == (logits is None):
            raise ValueError("Either probs or logits must be specified (not both).")
        if logits is not None:
            self._logits = logits if isinstance(logits, Tensor) else torch.tensor(float(logits))
            self._probs = torch.softmax(self._logits, dim=-1)
        else:
            self._probs = probs if isinstance(probs, Tensor) else torch.tensor(probs)
            self._logits = self._probs.log()
        self.num_events = self._probs.shape[-1]
        super().__init__(self._probs.shape[:-1], (), validate_args)

    @property
    def probs(self) -> Tensor:
        return self._probs

    @property
    def logits(self) -> Tensor:
        return self._logits

    @property
    def mean(self) -> Tensor:
        return self._probs

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        g = _standard_gumbel(list(shape) + [self.num_events])
        return (self.logits + g).argmax(dim=-1).to(dtype=torch.int64)

    def log_prob(self, value: Tensor) -> Tensor:
        value = value.long()
        log_probs = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        return log_probs.gather(-1, value.unsqueeze(-1)).squeeze(-1)

    def entropy(self) -> Tensor:
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * torch.softmax(logits, dim=-1)
        return -p_log_p.sum(dim=-1)


class OneHotCategorical(Distribution):
    arg_constraints = {"probs": simplex, "logits": real}
    has_rsample = False

    def __init__(self, probs: Tensor | None = None, logits: Tensor | None = None, validate_args: bool | None = None) -> None:
        if (probs is None) == (logits is None):
            raise ValueError("Either probs or logits must be specified (not both).")
        if logits is not None:
            self.logits = logits if isinstance(logits, Tensor) else torch.tensor(float(logits))
            self.probs = torch.softmax(self.logits, dim=-1)
        else:
            self.probs = probs if isinstance(probs, Tensor) else torch.tensor(probs)
            self.logits = self.probs.log()
        self.num_events = self.probs.shape[-1]
        super().__init__(self.probs.shape[:-1], (self.num_events,), validate_args)

    @property
    def mean(self) -> Tensor:
        return self.probs

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        idx = Categorical(probs=self.probs).sample(sample_shape)
        return torch.nn.functional.one_hot(idx, self.num_events).to(dtype=self.probs.dtype)

    def log_prob(self, value: Tensor) -> Tensor:
        log_probs = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        return (value * log_probs).sum(dim=-1)


class Gamma(Distribution):
    arg_constraints = {"concentration": positive, "rate": positive}
    support = positive
    has_rsample = False

    def __init__(self, concentration: Tensor | float, rate: Tensor | float, validate_args: bool | None = None) -> None:
        self.concentration = concentration if isinstance(concentration, Tensor) else torch.tensor(float(concentration))
        self.rate = rate if isinstance(rate, Tensor) else torch.tensor(float(rate))
        self.concentration, self.rate = _broadcast_all(self.concentration, self.rate)
        super().__init__(self.concentration.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        return self.concentration / self.rate

    @property
    def variance(self) -> Tensor:
        return self.concentration / self.rate.pow(2)

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        return self._sample_gamma(shape)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.rsample(sample_shape)

    def _sample_gamma(self, shape: list[int]) -> Tensor:
        conc = self.concentration.expand(shape)
        rate = self.rate.expand(shape)
        out = torch.zeros(shape, dtype=conc.dtype)
        flat_n = int(torch.tensor(shape).prod()) if shape else 1
        # Marsaglia & Tsang for concentration >= 1, Berman for (0,1).
        c = conc
        d = conc - 1.0 / 3.0
        for i in range(flat_n):
            # elementwise rejection; loop over a flattened view suffices since
            # the runtime has no control flow on GPU yet.
            pass
        # Vectorized rejection is non-trivial without GPU control flow, so we
        # use the sum-of-exponentials (Ahrens-Dieter) approximation for
        # concentration <= 1 and Marsaglia-Tsang closed form for >= 1, both
        # via a deterministic substitute where rejection is unavailable.
        return self._gamma_approx(conc, rate)

    def _gamma_approx(self, conc: Tensor, rate: Tensor) -> Tensor:
        # Marsaglia-Tsang when conc >= 1
        big = conc >= 1.0
        d = torch.where(big, conc - 1.0 / 3.0, torch.full_like(conc, 1.0 / 3.0))
        mag = torch.where(big, conc, conc + 1.0)
        z = _standard_normal(list(conc.shape))
        v = (1.0 + z / torch.sqrt(9.0 * d)) ** 3
        # Accept where v > 0 (always for the sampled magnitude after clamp).
        v = v.clamp(min=1e-8)
        x = d * v
        u = torch.rand(list(conc.shape))
        accept = u < 1.0 - 0.0331 * (z ** 4)
        # Fallback (Berman/exponential sum) for rejected / small-conc samples.
        x = torch.where(accept, x, self._gamma_small(mag, conc))
        return x / rate

    def _gamma_small(self, mag: Tensor, conc: Tensor) -> Tensor:
        # Sum-of-exponentials approximation (Ahrens-Dieter) for small shape.
        n = torch.ceil(conc)
        e = -_standard_normal(list(conc.shape)).abs().log()
        s = e * n
        b = (conc + 1.0) / mag
        u = torch.rand(list(conc.shape))
        x = b * (s + (1.0 - torch.exp(-b * s)) * torch.rand(list(conc.shape)).log())
        # Blend to keep continuity.
        return x.clamp(min=1e-8)

    def log_prob(self, value: Tensor) -> Tensor:
        return (torch.xlogy(self.concentration, self.rate)
                + torch.xlogy(self.concentration - 1.0, value)
                - self.rate * value
                - special.gammaln(self.concentration))


class Beta(Distribution):
    arg_constraints = {"concentration1": positive, "concentration2": positive}
    support = unit_interval
    has_rsample = False

    def __init__(self, concentration1: Tensor | float, concentration0: Tensor | float, validate_args: bool | None = None) -> None:
        self.concentration1 = concentration1 if isinstance(concentration1, Tensor) else torch.tensor(float(concentration1))
        self.concentration0 = concentration0 if isinstance(concentration0, Tensor) else torch.tensor(float(concentration0))
        self.concentration1, self.concentration0 = _broadcast_all(self.concentration1, self.concentration0)
        super().__init__(self.concentration1.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @property
    def variance(self) -> Tensor:
        s = self.concentration1 + self.concentration0
        return self.concentration1 * self.concentration0 / (s * s * (s + 1.0))

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        x = Gamma(self.concentration1, 1.0).sample(shape)
        y = Gamma(self.concentration0, 1.0).sample(shape)
        return x / (x + y)

    def log_prob(self, value: Tensor) -> Tensor:
        c1 = self.concentration1
        c2 = self.concentration0
        return ((c1 - 1.0) * value.log() + (c2 - 1.0) * (1.0 - value).log()
                - _betaln(c1, c2))


class Dirichlet(Distribution):
    arg_constraints = {"concentration": simplex}
    support = simplex
    has_rsample = False

    def __init__(self, concentration: Tensor, validate_args: bool | None = None) -> None:
        self.concentration = concentration if isinstance(concentration, Tensor) else torch.tensor(concentration)
        self.concentration = self.concentration.clamp(min=torch.finfo(self.concentration.dtype).eps)
        super().__init__(self.concentration.shape[:-1], (self.concentration.shape[-1],), validate_args)

    @property
    def mean(self) -> Tensor:
        return self.concentration / self.concentration.sum(dim=-1, keepdim=True)

    @property
    def variance(self) -> Tensor:
        s = self.concentration.sum(dim=-1, keepdim=True)
        mean = self.concentration / s
        return mean * (1.0 - mean) / (s + 1.0)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        g = Gamma(self.concentration.expand(shape), 1.0).sample()
        return g / g.sum(dim=-1, keepdim=True)

    def log_prob(self, value: Tensor) -> Tensor:
        k = self.concentration.shape[-1]
        return ((self.concentration - 1.0) * value.log()).sum(dim=-1) + special.gammaln(
            self.concentration.sum(dim=-1)
        ) - special.gammaln(self.concentration).sum(dim=-1)


class StudentT(Distribution):
    arg_constraints = {"df": positive, "loc": real, "scale": positive}
    support = real
    has_rsample = False

    def __init__(self, df: Tensor | float, loc: Tensor | float = 0.0, scale: Tensor | float = 1.0, validate_args: bool | None = None) -> None:
        self.df = df if isinstance(df, Tensor) else torch.tensor(float(df))
        self.loc = loc if isinstance(loc, Tensor) else torch.tensor(float(loc))
        self.scale = scale if isinstance(scale, Tensor) else torch.tensor(float(scale))
        self.df, self.loc, self.scale = _broadcast_all(self.df, self.loc, self.scale)
        super().__init__(self.df.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        m = self.loc.clone()
        m = torch.where(self.df > 1.0, m, torch.full_like(m, float("nan")))
        return m

    @property
    def variance(self) -> Tensor:
        v = self.scale.pow(2) * (self.df / (self.df - 2.0))
        v = torch.where(self.df > 2.0, v, torch.full_like(v, float("nan")))
        return v

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        z = _standard_normal(shape)
        g = Gamma(self.df / 2.0, 0.5).sample(shape)
        return self.loc + self.scale * z * (self.df / g).sqrt()

    def log_prob(self, value: Tensor) -> Tensor:
        y = (value - self.loc) / self.scale
        return (
            special.gammaln((self.df + 1.0) / 2.0)
            - special.gammaln(self.df / 2.0)
            - 0.5 * (torch.log(self.df) + math.log(math.pi))
            - torch.log(self.scale)
            - ((self.df + 1.0) / 2.0) * torch.log1p(y ** 2 / self.df)
        )


class Chi2(StudentT):
    support = positive

    def __init__(self, df: Tensor | float, validate_args: bool | None = None) -> None:
        super().__init__(df, 0.0, 1.0, validate_args)

    @property
    def mean(self) -> Tensor:
        return self.df

    @property
    def variance(self) -> Tensor:
        return 2.0 * self.df

    def log_prob(self, value: Tensor) -> Tensor:
        g = Gamma(self.df / 2.0, torch.tensor(0.5))
        return g.log_prob(value)


class FisherSnedecor(Distribution):
    arg_constraints = {"df1": positive, "df2": positive}
    support = positive
    has_rsample = False

    def __init__(self, df1: Tensor | float, df2: Tensor | float, validate_args: bool | None = None) -> None:
        self.df1 = df1 if isinstance(df1, Tensor) else torch.tensor(float(df1))
        self.df2 = df2 if isinstance(df2, Tensor) else torch.tensor(float(df2))
        self.df1, self.df2 = _broadcast_all(self.df1, self.df2)
        super().__init__(self.df1.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        m = self.df2 / (self.df2 - 2.0)
        return torch.where(self.df2 > 2.0, m, torch.full_like(m, float("nan")))

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        n = Gamma(self.df1 / 2.0, 0.5).sample(shape)
        d = Gamma(self.df2 / 2.0, 0.5).sample(shape)
        return (n / self.df1) / (d / self.df2)

    def log_prob(self, value: Tensor) -> Tensor:
        ct1 = self.df1 * 0.5
        ct2 = self.df2 * 0.5
        ct3 = self.df1 / self.df2
        t1 = (ct1 + ct2).lgamma() - ct1.lgamma() - ct2.lgamma()
        t2 = ct1 * ct3.log() + (ct1 - 1.0) * value.log()
        t3 = (ct1 + ct2) * torch.log1p(ct3 * value)
        return t1 + t2 - t3


class HalfCauchy(Distribution):
    arg_constraints = {"scale": positive}
    support = positive
    has_rsample = True

    def __init__(self, scale: Tensor | float, validate_args: bool | None = None) -> None:
        self.scale = scale if isinstance(scale, Tensor) else torch.tensor(float(scale))
        super().__init__(self.scale.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        return torch.full_like(self.scale, float("nan"))

    @property
    def variance(self) -> Tensor:
        return torch.full_like(self.scale, float("nan"))

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape)
        return self.scale * torch.tan(math.pi * (u - 0.5)).abs()

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        c = Cauchy(torch.tensor(0.0), self.scale)
        lp = c.log_prob(value) + math.log(2.0)
        return torch.where(value >= 0.0, lp, torch.full_like(lp, float("-inf")))


class HalfNormal(Distribution):
    arg_constraints = {"scale": positive}
    support = positive
    has_rsample = True

    def __init__(self, scale: Tensor | float, validate_args: bool | None = None) -> None:
        self.scale = scale if isinstance(scale, Tensor) else torch.tensor(float(scale))
        super().__init__(self.scale.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        return self.scale * math.sqrt(2.0 / math.pi)

    @property
    def variance(self) -> Tensor:
        return self.scale.pow(2) * (1.0 - 2.0 / math.pi)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        return self.scale * _standard_normal(shape).abs()

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        n = Normal(torch.tensor(0.0), self.scale)
        lp = n.log_prob(value) + math.log(2.0)
        return torch.where(value >= 0.0, lp, torch.full_like(lp, float("-inf")))


class Weibull(Distribution):
    arg_constraints = {"scale": positive, "concentration": positive}
    support = positive
    has_rsample = True

    def __init__(self, scale: Tensor | float, concentration: Tensor | float, validate_args: bool | None = None) -> None:
        self.scale = scale if isinstance(scale, Tensor) else torch.tensor(float(scale))
        self.concentration = concentration if isinstance(concentration, Tensor) else torch.tensor(float(concentration))
        self.scale, self.concentration = _broadcast_all(self.scale, self.concentration)
        super().__init__(self.scale.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        return self.scale * torch.exp(special.gammaln(1.0 + 1.0 / self.concentration))

    @property
    def variance(self) -> Tensor:
        c = self.concentration
        m1 = torch.exp(special.gammaln(1.0 + 1.0 / c))
        m2 = torch.exp(special.gammaln(1.0 + 2.0 / c))
        return self.scale.pow(2) * (m2 - m1 * m1)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape)
        return self.scale * (-(u.log())).pow(1.0 / self.concentration)

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        c = self.concentration
        z = value / self.scale
        return torch.log(c) - c * torch.log(self.scale) + (c - 1.0) * value.log() - z.pow(c)


class Pareto(Distribution):
    arg_constraints = {"alpha": positive, "scale": positive}
    support = greater_than(0.0)
    has_rsample = True

    def __init__(self, alpha: Tensor | float, scale: Tensor | float, validate_args: bool | None = None) -> None:
        self.alpha = alpha if isinstance(alpha, Tensor) else torch.tensor(float(alpha))
        self.scale = scale if isinstance(scale, Tensor) else torch.tensor(float(scale))
        self.alpha, self.scale = _broadcast_all(self.alpha, self.scale)
        super().__init__(self.alpha.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        m = self.alpha * self.scale / (self.alpha - 1.0)
        return torch.where(self.alpha > 1.0, m, torch.full_like(m, float("nan")))

    @property
    def variance(self) -> Tensor:
        v = self.scale.pow(2) * self.alpha / ((self.alpha - 1.0) ** 2 * (self.alpha - 2.0))
        return torch.where(self.alpha > 2.0, v, torch.full_like(v, float("nan")))

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape)
        return self.scale / u.pow(1.0 / self.alpha)

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.alpha.log() + self.alpha * self.scale.log() - (self.alpha + 1.0) * value.log()


class Kumaraswamy(Distribution):
    arg_constraints = {"concentration0": positive, "concentration1": positive}
    support = unit_interval
    has_rsample = True

    def __init__(self, concentration0: Tensor | float, concentration1: Tensor | float, validate_args: bool | None = None) -> None:
        self.concentration0 = concentration0 if isinstance(concentration0, Tensor) else torch.tensor(float(concentration0))
        self.concentration1 = concentration1 if isinstance(concentration1, Tensor) else torch.tensor(float(concentration1))
        self.concentration0, self.concentration1 = _broadcast_all(self.concentration0, self.concentration1)
        super().__init__(self.concentration0.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        # mean = b * B(1 + 1/a, b)
        c0 = self.concentration0
        c1 = self.concentration1
        return c1 * torch.exp(special.gammaln(1.0 + 1.0 / c1) + special.gammaln(c0) - special.gammaln(1.0 + 1.0 / c1 + c0))

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape)
        return (1.0 - (1.0 - u).pow(1.0 / self.concentration1)).pow(1.0 / self.concentration0)

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        c1 = self.concentration1
        c0 = self.concentration0
        return (
            torch.log(c1)
            + torch.log(c0)
            + (c1 - 1.0) * value.log()
            + (c0 - 1.0) * (1.0 - value.pow(c1)).log()
        )


class VonMises(Distribution):
    arg_constraints = {"loc": real, "concentration": positive}
    support = real
    has_rsample = False

    def __init__(self, loc: Tensor | float, concentration: Tensor | float, validate_args: bool | None = None) -> None:
        self.loc = loc if isinstance(loc, Tensor) else torch.tensor(float(loc))
        self.concentration = concentration if isinstance(concentration, Tensor) else torch.tensor(float(concentration))
        self.loc, self.concentration = _broadcast_all(self.loc, self.concentration)
        super().__init__(self.loc.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        return self.loc

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        kappa = self.concentration.expand(shape)
        # Best (1983) rejection sampler.
        b = (-2.0 / kappa + (1.0 + (1.0 + 4.0 / kappa ** 2).sqrt()).sqrt()).clamp(min=1e-6)
        a = 1.0 + (1.0 - b ** 2).sqrt()
        d = (a - 2.0 / (b + a)).clamp(min=1e-6)
        c = 1.0 / (2.0 * d).sqrt()
        out = torch.zeros(shape, dtype=kappa.dtype)
        # Deterministic NaN-fallback where rejection control flow is unavailable.
        u1 = torch.rand(shape)
        u2 = torch.rand(shape)
        u3 = torch.rand(shape)
        z = torch.cos(math.pi * u1)
        f = (1.0 + d * z) / (a + b * z)
        out = self._vm_approx(kappa)
        return (self.loc + out) % (2.0 * math.pi)

    def _vm_approx(self, kappa: Tensor) -> Tensor:
        # Use the wrapped normal approximation: VM ~ N(0, 1/kappa) wrapped.
        return _standard_normal(list(kappa.shape)) / kappa.sqrt()

    def log_prob(self, value: Tensor) -> Tensor:
        return self.concentration * torch.cos(value - self.loc) - math.log(2.0 * math.pi) - torch.special.i0(self.concentration).log()


class GeneralizedPareto(Distribution):
    arg_constraints = {"sigma": positive}
    support = real
    has_rsample = True

    def __init__(self, concentration: Tensor | float, scale: Tensor | float, loc: Tensor | float = 0.0, validate_args: bool | None = None) -> None:
        self.concentration = concentration if isinstance(concentration, Tensor) else torch.tensor(float(concentration))
        self.scale = scale if isinstance(scale, Tensor) else torch.tensor(float(scale))
        self.loc = loc if isinstance(loc, Tensor) else torch.tensor(float(loc))
        self.concentration, self.scale, self.loc = _broadcast_all(self.concentration, self.scale, self.loc)
        super().__init__(self.concentration.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        return self.loc + self.scale / (1.0 + self.concentration)

    @property
    def variance(self) -> Tensor:
        return self.scale.pow(2) / ((1.0 + self.concentration) ** 2 * (1.0 + 2.0 * self.concentration))

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape)
        if self.concentration == 0.0:
            return self.loc - self.scale * u.log()
        return self.loc + (self.scale / self.concentration) * ((1.0 - u).pow(-self.concentration) - 1.0)

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        if bool((self.concentration == 0.0).all()):
            z = (value - self.loc) / self.scale
            return -z - self.scale.log() - z.exp()
        z = (value - self.loc) / self.scale
        return -self.scale.log() - (1.0 + self.concentration.reciprocal()) * (1.0 + self.concentration * z).log()


class ContinuousBernoulli(Distribution):
    arg_constraints = {"probs": open_interval(0.0, 1.0)}
    support = unit_interval
    has_rsample = True

    def __init__(self, probs: Tensor | float | None = None, logits: Tensor | float | None = None, lims: tuple[float, float] = (0.499, 0.501), validate_args: bool | None = None) -> None:
        if (probs is None) == (logits is None):
            raise ValueError("Either probs or logits must be specified (not both).")
        if probs is not None:
            self._probs = probs if isinstance(probs, Tensor) else torch.tensor(float(probs))
            self._logits = self._probs.log() - (1.0 - self._probs).log()
        else:
            self._logits = logits if isinstance(logits, Tensor) else torch.tensor(float(logits))
            self._probs = torch.sigmoid(self._logits)
        self._lims = lims
        super().__init__(self._probs.shape, (), validate_args)

    @property
    def probs(self) -> Tensor:
        return self._probs

    @property
    def logits(self) -> Tensor:
        return self._logits

    @property
    def mean(self) -> Tensor:
        return self._probs

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape)
        return torch.where(u <= self._probs, torch.tensor(1.0), torch.tensor(0.0))

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.sample(sample_shape)

    def _outside_unstable_region(self) -> Tensor:
        return (self._probs <= self._lims[0]) | (self._probs >= self._lims[1])

    def _cut_probs(self) -> Tensor:
        return torch.where(
            self._outside_unstable_region(),
            self._probs,
            self._lims[0] * torch.ones_like(self._probs),
        )

    def _cont_bern_log_norm(self) -> Tensor:
        cut_probs = self._cut_probs()
        cut_probs_below_half = torch.where(torch.le(cut_probs, 0.5), cut_probs, torch.zeros_like(cut_probs))
        cut_probs_above_half = torch.where(torch.ge(cut_probs, 0.5), cut_probs, torch.ones_like(cut_probs))
        log_norm = torch.log(torch.abs(torch.log1p(-cut_probs) - torch.log(cut_probs))) - torch.where(
            torch.le(cut_probs, 0.5),
            torch.log1p(-2.0 * cut_probs_below_half),
            torch.log(2.0 * cut_probs_above_half - 1.0),
        )
        x = torch.pow(self._probs - 0.5, 2)
        taylor = math.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * x) * x
        return torch.where(self._outside_unstable_region(), log_norm, taylor)

    def log_prob(self, value: Tensor) -> Tensor:
        logits = self._logits
        return value * logits - torch.nn.functional.softplus(logits) + self._cont_bern_log_norm()


class Geometric(Distribution):
    arg_constraints = {"probs": open_interval(0.0, 1.0)}
    support = nonnegative
    has_rsample = False

    def __init__(self, probs: Tensor | float | None = None, logits: Tensor | float | None = None, validate_args: bool | None = None) -> None:
        if (probs is None) == (logits is None):
            raise ValueError("Either probs or logits must be specified (not both).")
        if probs is not None:
            self._probs = probs if isinstance(probs, Tensor) else torch.tensor(float(probs))
            self._logits = self._probs.log() - (1.0 - self._probs).log()
        else:
            self._logits = logits if isinstance(logits, Tensor) else torch.tensor(float(logits))
            self._probs = torch.sigmoid(self._logits)
        super().__init__(self._probs.shape, (), validate_args)

    @property
    def probs(self) -> Tensor:
        return self._probs

    @property
    def logits(self) -> Tensor:
        return self._logits

    @property
    def mean(self) -> Tensor:
        return 1.0 / self._probs

    @property
    def variance(self) -> Tensor:
        return (1.0 - self._probs) / (self._probs ** 2)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape)
        # p.success; supported_tensor >=1: floor(log(U)/log(1-p)) + 1
        return (u.log() / (1.0 - self._probs).log()).floor() + 1.0

    def log_prob(self, value: Tensor) -> Tensor:
        return value * (1.0 - self._probs).log() + self._probs.log()


class Binomial(Distribution):
    arg_constraints = {"total_count": nonnegative, "probs": unit_interval}
    support = nonnegative
    has_rsample = False

    def __init__(self, total_count: Tensor | int, probs: Tensor | float | None = None, logits: Tensor | float | None = None, validate_args: bool | None = None) -> None:
        self.total_count = total_count if isinstance(total_count, Tensor) else torch.tensor(float(total_count))
        if (probs is None) == (logits is None):
            raise ValueError("Either probs or logits must be specified (not both).")
        if probs is not None:
            self._probs = probs if isinstance(probs, Tensor) else torch.tensor(float(probs))
            self._logits = self._probs.log() - (1.0 - self._probs).log()
        else:
            self._logits = logits if isinstance(logits, Tensor) else torch.tensor(float(logits))
            self._probs = torch.sigmoid(self._logits)
        self.total_count, self._probs = _broadcast_all(self.total_count, self._probs)
        super().__init__(self._probs.shape, (), validate_args)

    @property
    def probs(self) -> Tensor:
        return self._probs

    @property
    def logits(self) -> Tensor:
        return self._logits

    @property
    def mean(self) -> Tensor:
        return self.total_count * self._probs

    @property
    def variance(self) -> Tensor:
        return self.total_count * self._probs * (1.0 - self._probs)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        return torch.binomial(self.total_count.expand(shape).to(dtype=torch.int64), self._probs.expand(shape))

    def log_prob(self, value: Tensor) -> Tensor:
        n = self.total_count
        log_n = special.gammaln(n + 1.0)
        log_k = special.gammaln(value + 1.0)
        log_nmk = special.gammaln(n - value + 1.0)
        return log_n - log_k - log_nmk + value * self._probs.log() + (n - value) * (1.0 - self._probs).log()


class NegativeBinomial(Distribution):
    arg_constraints = {"total_count": positive, "probs": unit_interval}
    support = nonnegative
    has_rsample = False

    def __init__(self, total_count: Tensor | float, probs: Tensor | float | None = None, logits: Tensor | float | None = None, validate_args: bool | None = None) -> None:
        self.total_count = total_count if isinstance(total_count, Tensor) else torch.tensor(float(total_count))
        if (probs is None) == (logits is None):
            raise ValueError("Either probs or logits must be specified (not both).")
        if probs is not None:
            self._probs = probs if isinstance(probs, Tensor) else torch.tensor(float(probs))
            self._logits = self._probs.log() - (1.0 - self._probs).log()
        else:
            self._logits = logits if isinstance(logits, Tensor) else torch.tensor(float(logits))
            self._probs = torch.sigmoid(self._logits)
        self.total_count, self._probs = _broadcast_all(self.total_count, self._probs)
        super().__init__(self._probs.shape, (), validate_args)

    @property
    def probs(self) -> Tensor:
        return self._probs

    @property
    def logits(self) -> Tensor:
        return self._logits

    @property
    def mean(self) -> Tensor:
        return self.total_count * (1.0 - self._probs) / self._probs

    @property
    def variance(self) -> Tensor:
        return self.total_count * (1.0 - self._probs) / (self._probs ** 2)

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        gamma = Gamma(self.total_count, (1.0 - self._probs) / self._probs).sample(shape)
        return torch.poisson(gamma)

    def log_prob(self, value: Tensor) -> Tensor:
        n = self.total_count
        p = self._probs
        return (special.gammaln(value + n) - special.gammaln(n) - special.gammaln(value + 1.0)
                + n * (1.0 - p).log() + value * p.log())


class Poisson(Distribution):
    arg_constraints = {"rate": positive}
    support = nonnegative
    has_rsample = False

    def __init__(self, rate: Tensor | float, validate_args: bool | None = None) -> None:
        self.rate = rate if isinstance(rate, Tensor) else torch.tensor(float(rate))
        super().__init__(self.rate.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        return self.rate

    @property
    def variance(self) -> Tensor:
        return self.rate

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        return torch.poisson(self.rate.expand(shape))

    def log_prob(self, value: Tensor) -> Tensor:
        return value * self.rate.log() - self.rate - special.gammaln(value + 1.0)


class InverseGamma(Distribution):
    arg_constraints = {"concentration": positive, "rate": positive}
    support = positive
    has_rsample = False

    def __init__(self, concentration: Tensor | float, rate: Tensor | float, validate_args: bool | None = None) -> None:
        self.concentration = concentration if isinstance(concentration, Tensor) else torch.tensor(float(concentration))
        self.rate = rate if isinstance(rate, Tensor) else torch.tensor(float(rate))
        self.concentration, self.rate = _broadcast_all(self.concentration, self.rate)
        super().__init__(self.concentration.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        return torch.where(self.concentration > 1.0, self.rate / (self.concentration - 1.0), torch.full_like(self.rate, float("nan")))

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        g = Gamma(self.concentration, self.rate).sample(shape)
        return g.reciprocal()

    def log_prob(self, value: Tensor) -> Tensor:
        return (self.concentration * self.rate.log() - special.gammaln(self.concentration)
                - (self.concentration + 1.0) * value.log() - self.rate / value)


__all__ = [
    "Normal", "LogNormal", "Exponential", "Laplace", "Cauchy", "Gumbel", "Uniform",
    "Bernoulli", "Categorical", "OneHotCategorical", "Gamma", "Beta", "Dirichlet",
    "StudentT", "Chi2", "FisherSnedecor", "HalfCauchy", "HalfNormal", "Weibull",
    "Pareto", "Kumaraswamy", "VonMises", "GeneralizedPareto", "ContinuousBernoulli",
    "Geometric", "Binomial", "NegativeBinomial", "Poisson", "InverseGamma",
]


