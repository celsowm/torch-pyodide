"""Composition and relaxed distributions: Independent, TransformedDistribution,
ExponentialFamily, RelaxedBernoulli, RelaxedOneHotCategorical.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor

from .utils import Distribution, _as_shape, _standard_gumbel, _broadcast_all
from .constraints import real, unit_interval, simplex, real_vector, Constraint
from .transforms import Transform, SigmoidTransform, ExpTransform, ComposeTransform
from .univariate import Bernoulli, Categorical


class Independent(Distribution):
    """Reinterpret the last ``reinterpreted_batch_ndims`` batch dims of a base
    distribution as event dims."""

    has_rsample = False

    def __init__(self, base_distribution: Distribution, reinterpreted_batch_ndims: int, validate_args: bool | None = None) -> None:
        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        batch_shape = list(base_distribution.batch_shape[:-reinterpreted_batch_ndims]) if base_distribution.batch_shape else []
        event_shape = list(base_distribution.batch_shape[-reinterpreted_batch_ndims:]) + list(base_distribution.event_shape)
        super().__init__(batch_shape, event_shape, validate_args)
        self.has_rsample = base_distribution.has_rsample

    @property
    def mean(self) -> Tensor:
        return self.base_dist.mean

    @property
    def variance(self) -> Tensor:
        return self.base_dist.variance

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        lp = self.base_dist.log_prob(value)
        if self.reinterpreted_batch_ndims == 0:
            return lp
        for d in range(len(lp.shape) - self.reinterpreted_batch_ndims, len(lp.shape)):
            lp = lp.sum(dim=d)
        return lp


def _sum_rightmost(value: Tensor, n: int) -> Tensor:
    if n <= 0:
        return value
    result = value
    for _ in range(n):
        result = result.sum(dim=-1)
    return result


class TransformedDistribution(Distribution):
    """A distribution transformed by a sequence of ``Transform`` objects."""

    has_rsample = False

    def __init__(self, base_distribution: Distribution, transforms: list[Transform], validate_args: bool | None = None) -> None:
        if isinstance(transforms, Transform):
            self.transforms = [transforms]
        elif isinstance(transforms, list):
            if not all(isinstance(t, Transform) for t in transforms):
                raise ValueError("transforms must be a Transform or a list of Transforms")
            self.transforms = transforms
        else:
            raise ValueError("transforms must be a Transform or list, but was {0}".format(transforms))

        base_distribution = base_distribution
        base_shape = list(base_distribution.batch_shape) + list(base_distribution.event_shape)
        base_event_dim = len(base_distribution.event_shape)
        transform = ComposeTransform(self.transforms)
        if len(base_shape) < transform.domain.event_dim:
            raise ValueError(
                "base_distribution needs to have shape with size at least {0}, but got {1}.".format(transform.domain.event_dim, base_shape)
            )
        forward_shape = transform.forward_shape(base_shape)
        expanded_base_shape = transform.inverse_shape(forward_shape)
        if base_shape != expanded_base_shape:
            base_batch_shape = expanded_base_shape[: len(expanded_base_shape) - base_event_dim]
            base_distribution = base_distribution.expand(base_batch_shape)
        reinterpreted_batch_ndims = transform.domain.event_dim - base_event_dim
        if reinterpreted_batch_ndims > 0:
            base_distribution = Independent(base_distribution, reinterpreted_batch_ndims)
        self.base_dist = base_distribution

        transform_change_in_event_dim = transform.codomain.event_dim - transform.domain.event_dim
        event_dim = max(transform.codomain.event_dim, base_event_dim + transform_change_in_event_dim)
        if len(forward_shape) < event_dim:
            raise AssertionError("forward_shape length {0} must be >= event_dim {1}".format(len(forward_shape), event_dim))
        cut = len(forward_shape) - event_dim
        batch_shape = forward_shape[:cut]
        event_shape = forward_shape[cut:]
        super().__init__(batch_shape, event_shape, validate_args)
        self.has_rsample = base_distribution.has_rsample

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        x = self.base_dist.sample(sample_shape)
        for t in self.transforms:
            x = t(x)
        return x

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        x = self.base_dist.rsample(sample_shape)
        for t in self.transforms:
            x = t(x)
        return x

    def log_prob(self, value: Tensor) -> Tensor:
        if self.validate_args:
            self._validate_sample(value)
        event_dim = len(self.event_shape)
        log_prob: Tensor = torch.tensor(0.0)
        y = value
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            event_dim += transform.domain.event_dim - transform.codomain.event_dim
            log_prob = log_prob - _sum_rightmost(
                transform.log_abs_det_jacobian(x, y),
                event_dim - transform.domain.event_dim,
            )
            y = x
        log_prob = log_prob + _sum_rightmost(
            self.base_dist.log_prob(y),
            event_dim - len(self.base_dist.event_shape),
        )
        return log_prob


class ExponentialFamily(Distribution):
    """Mixin base for exponential-family distributions. Provides the standard
    ``entropy()`` via ``kl_divergence(self || base)`` and a ``_log_normalizer``
    hook. Concrete families supply ``natural_param`` / ``log_normalizer``."""

    def natural_param(self) -> Tensor:
        raise NotImplementedError

    def log_normalizer(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def entropy(self) -> Tensor:
        raise NotImplementedError


class LogitRelaxedBernoulli(Distribution):
    arg_constraints = {"probs": unit_interval, "logits": real}
    support = real

    def __init__(self, temperature: Tensor | float, probs: Tensor | float | None = None, logits: Tensor | float | None = None, validate_args: bool | None = None) -> None:
        if (probs is None) == (logits is None):
            raise ValueError("Either probs or logits must be specified (not both).")
        self.temperature = temperature if isinstance(temperature, Tensor) else torch.tensor(float(temperature))
        if probs is not None:
            self.probs = probs if isinstance(probs, Tensor) else torch.tensor(float(probs))
            self.logits = torch.logit(self.probs)
        else:
            self.logits = logits if isinstance(logits, Tensor) else torch.tensor(float(logits))
            self.probs = torch.sigmoid(self.logits)
        super().__init__(self.logits.shape, (), validate_args)

    @property
    def mean(self) -> Tensor:
        return self.probs

    def log_prob(self, value: Tensor) -> Tensor:
        logits, value = _broadcast_all(self.logits, value)
        diff = logits - value.mul(self.temperature)
        return self.temperature.log() + diff - 2.0 * diff.exp().log1p()


class RelaxedBernoulli(TransformedDistribution):
    arg_constraints = {"probs": unit_interval, "logits": real}
    support = unit_interval
    has_rsample = True

    def __init__(self, temperature: Tensor | float, probs: Tensor | float | None = None, logits: Tensor | float | None = None, validate_args: bool | None = None) -> None:
        base_dist = LogitRelaxedBernoulli(temperature, probs=probs, logits=logits)
        super().__init__(base_dist, SigmoidTransform(), validate_args=validate_args)

    @property
    def temperature(self) -> Tensor:
        return self.base_dist.temperature

    @property
    def logits(self) -> Tensor:
        return self.base_dist.logits

    @property
    def probs(self) -> Tensor:
        return self.base_dist.probs


class ExpRelaxedCategorical(Distribution):
    arg_constraints = {"probs": simplex, "logits": real_vector}
    support = real_vector
    has_rsample = True

    def __init__(self, temperature: Tensor | float, probs: Tensor | None = None, logits: Tensor | None = None, validate_args: bool | None = None) -> None:
        self.temperature = temperature if isinstance(temperature, Tensor) else torch.tensor(float(temperature))
        self._categorical = Categorical(probs=probs, logits=logits)
        batch_shape = self._categorical.batch_shape
        event_shape = (self._categorical.num_events,)
        self._num_events = self._categorical.num_events
        super().__init__(batch_shape, event_shape, validate_args)

    @property
    def logits(self) -> Tensor:
        return self._categorical.logits

    @property
    def probs(self) -> Tensor:
        return self._categorical.probs

    def log_prob(self, value: Tensor) -> Tensor:
        K = self._num_events
        logits, value = _broadcast_all(self.logits, value)
        log_scale = torch.full_like(self.temperature, float(K)).lgamma() - self.temperature.log().mul(-(K - 1))
        score = logits - value.mul(self.temperature)
        score = (score - score.logsumexp(dim=-1, keepdim=True)).sum(-1)
        return score + log_scale


class RelaxedOneHotCategorical(TransformedDistribution):
    arg_constraints = {"probs": simplex, "logits": real_vector}
    support = simplex
    has_rsample = True

    def __init__(self, temperature: Tensor | float, probs: Tensor | None = None, logits: Tensor | None = None, validate_args: bool | None = None) -> None:
        base_dist = ExpRelaxedCategorical(temperature, probs=probs, logits=logits, validate_args=validate_args)
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    @property
    def temperature(self) -> Tensor:
        return self.base_dist.temperature

    @property
    def logits(self) -> Tensor:
        return self.base_dist.logits

    @property
    def probs(self) -> Tensor:
        return self.base_dist.probs


__all__ = [
    "Independent",
    "TransformedDistribution",
    "ExponentialFamily",
    "RelaxedBernoulli",
    "RelaxedOneHotCategorical",
]
