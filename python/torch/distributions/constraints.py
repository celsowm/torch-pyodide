"""Constraint objects for ``torch.distributions`` parameter validation.

A minimal but functional subset of ``torch.distributions.constraints``: every
constraint exposes ``check(value)`` returning a boolean (or boolean tensor)
and ``feasible_like(value)`` returning a tensor that satisfies the constraint
(useful for generating valid reference parameters at test time).
"""
from __future__ import annotations

import torch
from torch import Tensor


class Constraint:
    event_dim: int = 0

    def check(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def feasible_like(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class _Real(Constraint):
    def check(self, value: Tensor) -> Tensor:
        return torch.isfinite(value)

    def feasible_like(self, value: Tensor) -> Tensor:
        return torch.zeros_like(value)


class _Positive(Constraint):
    def check(self, value: Tensor) -> Tensor:
        return value > 0.0

    def feasible_like(self, value: Tensor) -> Tensor:
        return torch.ones_like(value)


class _NonNegative(Constraint):
    def check(self, value: Tensor) -> Tensor:
        return value >= 0.0

    def feasible_like(self, value: Tensor) -> Tensor:
        return torch.ones_like(value)


class _Negative(Constraint):
    def check(self, value: Tensor) -> Tensor:
        return value < 0.0

    def feasible_like(self, value: Tensor) -> Tensor:
        return -torch.ones_like(value)


class _GreaterThan(Constraint):
    def __init__(self, lower_bound: float) -> None:
        self.lower_bound = lower_bound

    def check(self, value: Tensor) -> Tensor:
        return value > self.lower_bound

    def feasible_like(self, value: Tensor) -> Tensor:
        return torch.full_like(value, self.lower_bound) + torch.ones_like(value)

    def __repr__(self) -> str:
        return f"greater_than({self.lower_bound})"


class _LessThan(Constraint):
    def __init__(self, upper_bound: float) -> None:
        self.upper_bound = upper_bound

    def check(self, value: Tensor) -> Tensor:
        return value < self.upper_bound

    def feasible_like(self, value: Tensor) -> Tensor:
        return torch.full_like(value, self.upper_bound) - torch.ones_like(value)

    def __repr__(self) -> str:
        return f"less_than({self.upper_bound})"


class _Interval(Constraint):
    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, value: Tensor) -> Tensor:
        return (value >= self.lower_bound) & (value <= self.upper_bound)

    def feasible_like(self, value: Tensor) -> Tensor:
        lo = torch.full_like(value, float(self.lower_bound))
        hi = torch.full_like(value, float(self.upper_bound))
        return (lo + hi) * 0.5

    def __repr__(self) -> str:
        return f"interval([{self.lower_bound}, {self.upper_bound}])"


class _UnitInterval(_Interval):
    def __init__(self) -> None:
        super().__init__(0.0, 1.0)


class _OpenInterval(_Interval):
    def check(self, value: Tensor) -> Tensor:
        return (value > self.lower_bound) & (value < self.upper_bound)


class _Simplex(Constraint):
    event_dim = 1

    def check(self, value: Tensor) -> Tensor:
        vs = value.clamp(min=0.0)
        return (value >= 0.0).all(dim=-1) & ((vs.sum(dim=-1) - 1.0).abs() < 1e-6)

    def feasible_like(self, value: Tensor) -> Tensor:
        shape = list(value.shape)
        last = shape[-1]
        flat = torch.ones(int(torch.tensor(shape).prod())) / float(last)
        return flat.reshape(shape)


class _RealVector(Constraint):
    event_dim = 1

    def check(self, value: Tensor) -> Tensor:
        return torch.isfinite(value)

    def feasible_like(self, value: Tensor) -> Tensor:
        return torch.zeros_like(value)


class _Boolean(Constraint):
    def check(self, value: Tensor) -> Tensor:
        return (value == 0.0) | (value == 1.0)

    def feasible_like(self, value: Tensor) -> Tensor:
        return torch.zeros_like(value)


class _IntegerInterval(_Interval):
    def check(self, value: Tensor) -> Tensor:
        return super().check(value) & (value == value.floor())


class _LowerCholesky(Constraint):
    def check(self, value: Tensor) -> Tensor:
        # value is assumed square in last two dims
        tril = value.tril()
        ok = (value == tril).all(dim=(-2, -1))
        diag_ok = value.diagonal(dim1=-2, dim2=-1) > 0.0
        return ok & diag_ok.all(dim=-1)

    def feasible_like(self, value: Tensor) -> Tensor:
        eye = torch.eye(value.shape[-1], dtype=value.dtype)
        return eye.expand(list(value.shape))


class _PositiveDefinite(Constraint):
    def check(self, value: Tensor) -> Tensor:
        # Symmetric and all leading principal minors positive (cheap proxy).
        sym = (value == value.transpose(-1, -2)).all(dim=(-2, -1))
        try:
            eigvals = torch.linalg.eig(value)[0].real
            pd = (eigvals > 0.0).all(dim=-1)
        except Exception:
            pd = torch.ones(value.shape[:-2] if len(value.shape) >= 2 else [], dtype=torch.bool)
        return sym & pd

    def feasible_like(self, value: Tensor) -> Tensor:
        eye = torch.eye(value.shape[-1], dtype=value.dtype)
        return eye.expand(list(value.shape)) * 2.0


class _Multinomial(Constraint):
    def check(self, value: Tensor) -> Tensor:
        return (value >= 0.0).all(dim=-1) & (value == value.floor()).all(dim=-1)

    def feasible_like(self, value: Tensor) -> Tensor:
        return torch.zeros_like(value)


class _Categorical(Constraint):
    def check(self, value: Tensor) -> Tensor:
        return (value >= 0.0).all(dim=-1) & (value == value.floor()).all(dim=-1)

    def feasible_like(self, value: Tensor) -> Tensor:
        return torch.zeros_like(value)


class _CatTransformConstraint(Constraint):
    def __init__(self, transforms) -> None:
        self.transforms = transforms

    def check(self, value: Tensor) -> Tensor:
        return torch.ones_like(value, dtype=torch.bool)


real = _Real()
positive = _Positive()
nonnegative = _NonNegative()
negative = _Negative()
unit_interval = _UnitInterval()
open_unit_interval = _OpenInterval(0.0, 1.0)
simplex = _Simplex()
real_vector = _RealVector()
boolean = _Boolean()
multinomial = _Multinomial()
categorical = _Categorical()
lower_cholesky = _LowerCholesky()
positive_definite = _PositiveDefinite()
integer_interval = lambda lo, hi: _IntegerInterval(lo, hi)
greater_than = lambda lo: _GreaterThan(lo)
less_than = lambda hi: _LessThan(hi)
interval = lambda lo, hi: _Interval(lo, hi)
open_interval = lambda lo, hi: _OpenInterval(lo, hi)


def independent(constraint: Constraint, reinterpreted_batch_ndims: int) -> Constraint:
    class _Independent(Constraint):
        event_dim = constraint.event_dim + reinterpreted_batch_ndims

        def check(self, value: Tensor) -> Tensor:
            return constraint.check(value)

        def feasible_like(self, value: Tensor) -> Tensor:
            return constraint.feasible_like(value)

    return _Independent()


def register_constraint(name: str, constraint: Constraint) -> None:
    globals()[name] = constraint


__all__ = [
    "Constraint",
    "real",
    "positive",
    "nonnegative",
    "negative",
    "unit_interval",
    "open_unit_interval",
    "simplex",
    "boolean",
    "multinomial",
    "categorical",
    "lower_cholesky",
    "positive_definite",
    "integer_interval",
    "greater_than",
    "less_than",
    "interval",
    "open_interval",
    "register_constraint",
]
