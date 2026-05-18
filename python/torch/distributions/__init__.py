from __future__ import annotations

import math
import random

import torch
from torch import Tensor
from torch._tensor import tensor_from_data


class Distribution:
    def sample(self) -> Tensor:
        raise NotImplementedError

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError


class Normal(Distribution):
    def __init__(self, loc: Tensor | float, scale: Tensor | float) -> None:
        self.loc = loc if isinstance(loc, Tensor) else torch.tensor(float(loc))
        self.scale = scale if isinstance(scale, Tensor) else torch.tensor(float(scale))

    def sample(self, sample_shape: int | list[int] = ()) -> Tensor:
        shape = [sample_shape] if isinstance(sample_shape, int) else list(sample_shape)
        eps = torch.randn(shape)
        return self.loc + eps * self.scale

    def log_prob(self, value: Tensor) -> Tensor:
        var = self.scale * self.scale
        log_scale = self.scale.log()
        return -((value - self.loc) ** 2) / (2.0 * var) - log_scale - math.log(math.sqrt(2.0 * math.pi))


class Uniform(Distribution):
    def __init__(self, low: Tensor | float, high: Tensor | float) -> None:
        self.low = low if isinstance(low, Tensor) else torch.tensor(float(low))
        self.high = high if isinstance(high, Tensor) else torch.tensor(float(high))

    def sample(self, sample_shape: int | list[int] = ()) -> Tensor:
        shape = [sample_shape] if isinstance(sample_shape, int) else list(sample_shape)
        u = torch.rand(shape)
        return self.low + u * (self.high - self.low)

    def log_prob(self, value: Tensor) -> Tensor:
        log_width = (self.high - self.low).log()
        return torch.where((value >= self.low) & (value <= self.high), -log_width, torch.tensor(-float("inf")))


class Bernoulli(Distribution):
    def __init__(self, probs: Tensor | float) -> None:
        self.probs = probs if isinstance(probs, Tensor) else torch.tensor(float(probs))

    def sample(self, sample_shape: int | list[int] = ()) -> Tensor:
        shape = [sample_shape] if isinstance(sample_shape, int) else list(sample_shape)
        u = torch.rand(shape)
        return (u < self.probs).to(self.probs.dtype)

    def log_prob(self, value: Tensor) -> Tensor:
        return value * self.probs.log() + (1.0 - value) * (1.0 - self.probs).log()


class Categorical(Distribution):
    def __init__(self, logits: Tensor | None = None, probs: Tensor | None = None) -> None:
        if logits is not None:
            self.logits = logits
        elif probs is not None:
            self.logits = probs.log()
        else:
            raise ValueError("Either logits or probs must be provided")
        # Compute probs for sampling
        self._probs = torch.softmax(self.logits, dim=-1)

    def sample(self, sample_shape: int | list[int] = ()) -> Tensor:
        shape = [sample_shape] if isinstance(sample_shape, int) else list(sample_shape)
        n = 1
        for s in shape:
            n *= s
        # Gumbel-max trick
        gumbel = -(-torch.rand(list(shape) + list(self.logits.shape)).log()).log()
        return (self.logits + gumbel).argmax(dim=-1).reshape(shape + [1]).squeeze(-1)

    def log_prob(self, value: Tensor) -> Tensor:
        return torch.nn.functional.nll_loss(self.logits.log_softmax(dim=-1), value)


class Transforms:
    @staticmethod
    def sigmoid(x: Tensor) -> Tensor:
        return x.sigmoid()

    @staticmethod
    def log_sigmoid(x: Tensor) -> Tensor:
        return -(-x).expm1().log() - x.clamp(0.0)


__all__ = ["Distribution", "Normal", "Uniform", "Bernoulli", "Categorical", "Transforms"]
