"""``torch.distributions`` — a functional subset matching PyTorch's API.

Implements the full public class surface (distributions, transforms,
constraints) with ``sample`` / ``rsample`` and ``log_prob`` validated against
real PyTorch. Heavy random sampling runs on the WebGPU runtime.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.tensor_factories_ops import tensor_from_data

from .constraints import (
    Constraint,
    real,
    positive,
    nonnegative,
    negative,
    unit_interval,
    open_unit_interval,
    simplex,
    boolean,
    multinomial,
    categorical,
    lower_cholesky,
    positive_definite,
    integer_interval,
    greater_than,
    less_than,
    interval,
    open_interval,
    register_constraint,
)
from .transforms import (
    Transform,
    IdentityTransform,
    ExpTransform,
    LogTransform,
    PowerTransform,
    SigmoidTransform,
    TanhTransform,
    SoftmaxTransform,
    SoftplusTransform,
    AbsTransform,
    AffineTransform,
    ComposeTransform,
    CatTransform,
    StackTransform,
    ReshapeTransform,
    IndependentTransform,
    StickBreakingTransform,
    LowerCholeskyTransform,
    PositiveDefiniteTransform,
    CumulativeDistributionTransform,
    CorrCholeskyTransform,
)
from .utils import Distribution
from .univariate import (
    Normal,
    LogNormal,
    Exponential,
    Laplace,
    Cauchy,
    Gumbel,
    Uniform,
    Bernoulli,
    Categorical,
    OneHotCategorical,
    Gamma,
    Beta,
    Dirichlet,
    StudentT,
    Chi2,
    FisherSnedecor,
    HalfCauchy,
    HalfNormal,
    Weibull,
    Pareto,
    Kumaraswamy,
    VonMises,
    GeneralizedPareto,
    ContinuousBernoulli,
    Geometric,
    Binomial,
    NegativeBinomial,
    Poisson,
    InverseGamma,
)
from .multivariate import (
    MultivariateNormal,
    LowRankMultivariateNormal,
    LogisticNormal,
    Wishart,
    LKJCholesky,
    MixtureSameFamily,
)
from .distributions import (
    Independent,
    TransformedDistribution,
    ExponentialFamily,
    RelaxedBernoulli,
    RelaxedOneHotCategorical,
)

__all__ = [
    "Constraint", "real", "positive", "nonnegative", "negative", "unit_interval",
    "open_unit_interval", "simplex", "boolean", "multinomial", "categorical",
    "lower_cholesky", "positive_definite", "integer_interval", "greater_than",
    "less_than", "interval", "open_interval", "register_constraint",
    "Transform", "IdentityTransform", "ExpTransform", "LogTransform",
    "PowerTransform", "SigmoidTransform", "TanhTransform", "SoftmaxTransform",
    "SoftplusTransform", "AbsTransform", "AffineTransform", "ComposeTransform",
    "CatTransform", "StackTransform", "ReshapeTransform", "IndependentTransform",
    "StickBreakingTransform", "LowerCholeskyTransform", "PositiveDefiniteTransform",
    "CumulativeDistributionTransform", "CorrCholeskyTransform",
    "Distribution",
    "Normal", "LogNormal", "Exponential", "Laplace", "Cauchy", "Gumbel", "Uniform",
    "Bernoulli", "Categorical", "OneHotCategorical", "Gamma", "Beta", "Dirichlet",
    "StudentT", "Chi2", "FisherSnedecor", "HalfCauchy", "HalfNormal", "Weibull",
    "Pareto", "Kumaraswamy", "VonMises", "GeneralizedPareto", "ContinuousBernoulli",
    "Geometric", "Binomial", "NegativeBinomial", "Poisson", "InverseGamma",
    "MultivariateNormal", "LowRankMultivariateNormal", "LogisticNormal", "Wishart",
    "LKJCholesky", "MixtureSameFamily",
    "Independent", "TransformedDistribution", "ExponentialFamily",
    "RelaxedBernoulli", "RelaxedOneHotCategorical",
]


def kl_divergence(p: Distribution, q: Distribution) -> Tensor:
    """KL divergence for a few pairs; otherwise falls back to the definition
    ``E_p[log p - log q]`` via Monte-Carlo sampling when exact forms are
    unavailable."""
    try:
        return _kl_exact(p, q)
    except NotImplementedError:
        s = p.sample((1000,))
        return (p.log_prob(s) - q.log_prob(s)).mean(dim=0)


def _kl_exact(p: Distribution, q: Distribution) -> Tensor:
    if isinstance(p, Normal) and isinstance(q, Normal):
        var_ratio = (p.scale / q.scale) ** 2
        return ((p.loc - q.loc) ** 2) / (2.0 * q.scale ** 2) + 0.5 * (
            var_ratio - 1.0 - var_ratio.log()
        )
    if isinstance(p, Bernoulli) and isinstance(q, Bernoulli):
        return p.probs * (p.probs.log() - q.probs.log()) + (1.0 - p.probs) * (
            (1.0 - p.probs).log() - (1.0 - q.probs).log()
        )
    if isinstance(p, Categorical) and isinstance(q, Categorical):
        p_lp = p.logits - torch.logsumexp(p.logits, dim=-1, keepdim=True)
        q_lp = q.logits - torch.logsumexp(q.logits, dim=-1, keepdim=True)
        return (torch.softmax(p.logits, dim=-1) * (p_lp - q_lp)).sum(dim=-1)
    raise NotImplementedError


def _as_shape(sample_shape):
    return [sample_shape] if isinstance(sample_shape, int) else list(sample_shape)


def _broadcast_shape(*shapes):
    result = []
    max_rank = max((len(s) for s in shapes), default=0)
    for offset in range(max_rank):
        dims = [s[len(s) - max_rank + offset] if offset >= max_rank - len(s) else 1 for s in shapes]
        result.append(max(dims))
    return result
