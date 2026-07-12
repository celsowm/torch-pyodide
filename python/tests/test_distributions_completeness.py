from __future__ import annotations

import inspect

import torch.distributions as D

DISTRIBUTIONS = [
    "Normal", "Uniform", "Bernoulli", "Categorical", "OneHotCategorical",
    "Binomial", "Poisson", "Geometric", "NegativeBinomial", "Exponential",
    "Gamma", "Beta", "Dirichlet", "StudentT", "Cauchy", "Laplace", "Gumbel",
    "Pareto", "Weibull", "Chi2", "FisherSnedecor", "LogNormal",
    "HalfCauchy", "HalfNormal", "VonMises", "Kumaraswamy", "ContinuousBernoulli",
    "InverseGamma", "GeneralizedPareto", "MultivariateNormal",
    "LowRankMultivariateNormal", "LogisticNormal", "Wishart", "LKJCholesky",
    "Independent", "TransformedDistribution", "ExponentialFamily",
    "RelaxedBernoulli", "RelaxedOneHotCategorical", "MixtureSameFamily",
]

TRANSFORMS = [
    "AffineTransform", "SigmoidTransform", "TanhTransform", "ExpTransform",
    "LogTransform", "SoftmaxTransform", "StickBreakingTransform",
    "LowerCholeskyTransform", "CorrCholeskyTransform",
    "PositiveDefiniteTransform", "PowerTransform", "ComposeTransform",
]

CONSTRAINTS = [
    "Constraint", "real", "positive", "unit_interval", "simplex",
    "lower_cholesky", "positive_definite", "boolean", "nonnegative",
]


def test_distributions_present():
    for name in DISTRIBUTIONS:
        assert hasattr(D, name), f"torch.distributions.{name} is missing"
        assert isinstance(getattr(D, name), type), f"torch.distributions.{name} is not a class"


def test_transforms_present():
    for name in TRANSFORMS:
        assert hasattr(D, name), f"torch.distributions.{name} is missing"
        assert isinstance(getattr(D, name), type), f"torch.distributions.{name} is not a class"


def test_constraints_present():
    for name in CONSTRAINTS:
        assert hasattr(D.constraints, name), f"torch.distributions.constraints.{name} is missing"


def test_constructor_params_match_installed_torch():
    import json
    import os
    import subprocess
    import sys
    import tempfile

    script = r"""
import json, inspect
import torch.distributions as D
names = json.loads(input())
out = {}
for name in names:
    cls = getattr(D, name)
    out[name] = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
print(json.dumps(out))
"""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tempfile.gettempdir(),
        env=env,
        input=json.dumps(DISTRIBUTIONS),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        import pytest

        pytest.skip(f"installed PyTorch not importable: {proc.stderr.strip()}")

    upstream = json.loads(proc.stdout)
    # Full PyTorch-only constructor toggles that are optional to implement.
    optional = {"validate_args", "arg_constraints", "support", "reparameterized",
                "precision_matrix", "scale_tril", "lims"}
    for name in DISTRIBUTIONS:
        local = [
            p
            for p in inspect.signature(getattr(D, name).__init__).parameters
            if p != "self"
        ]
        if name == "MixtureSameFamily":
            # constructor is (component_distribution, mixture_distribution)
            pass
        missing = [p for p in upstream[name] if p not in local and p not in optional]
        assert not missing, f"torch.distributions.{name} missing params {missing}"
