"""Base Distribution class and sampling utilities for torch.distributions."""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor


def _as_shape(sample_shape) -> list[int]:
    if isinstance(sample_shape, int):
        return [sample_shape]
    return list(sample_shape)


def _broadcast_shapes(*shapes) -> list[int]:
    result: list[int] = []
    max_rank = max((len(s) for s in shapes), default=0)
    for offset in range(max_rank):
        dims = []
        for s in shapes:
            if offset >= max_rank - len(s):
                dims.append(s[len(s) - max_rank + offset])
            else:
                dims.append(1)
        size = max(dims)
        if any(d not in (1, size) for d in dims):
            raise ValueError(f"Shapes are not broadcastable: {shapes}")
        result.append(size)
    return result


def _broadcast_all(*tensors: Tensor) -> list[Tensor]:
    shapes = [tuple(t.shape) for t in tensors]
    shape = _broadcast_shapes(*shapes)
    return [t.expand(shape) for t in tensors]


def _standard_normal(shape: list[int]) -> Tensor:
    return torch.randn(shape)


def _standard_gumbel(shape: list[int]) -> Tensor:
    # Gumbel(0,1) = -log(-log(U)), U uniform in (0,1)
    u = torch.rand(shape)
    return -(-u.log()).log()


def _probs_to_logits(probs: Tensor, is_binary: bool = False) -> Tensor:
    ps = probs.clamp(min=torch.finfo(probs.dtype).eps, max=1.0 - torch.finfo(probs.dtype).eps)
    if is_binary:
        return (ps / (1.0 - ps)).log()
    return ps.log()


def _logits_to_probs(logits: Tensor, is_binary: bool = False) -> Tensor:
    if is_binary:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=-1)


class Distribution:
    has_rsample = False
    support = None
    arg_constraints: dict = {}

    def __init__(
        self,
        batch_shape: Iterable[int] = (),
        event_shape: Iterable[int] = (),
        validate_args: bool | None = None,
    ) -> None:
        self.batch_shape = list(batch_shape)
        self.event_shape = list(event_shape)
        self.validate_args = validate_args
        self._validate_sample_shape: bool = True

    def sample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        raise NotImplementedError

    def rsample(self, sample_shape: Iterable[int] = ()) -> Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} does not support rsample")

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def cdf(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def icdf(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    @property
    def mean(self) -> Tensor:
        raise NotImplementedError

    @property
    def variance(self) -> Tensor:
        raise NotImplementedError

    def expand(self, batch_shape: Iterable[int]) -> "Distribution":
        raise NotImplementedError

    def _extended_shape(self, sample_shape: Iterable[int] = ()) -> list[int]:
        return list(_as_shape(sample_shape)) + list(self.batch_shape) + list(self.event_shape)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(batch_shape={self.batch_shape})"
