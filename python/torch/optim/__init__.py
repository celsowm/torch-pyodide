from __future__ import annotations

from typing import Iterable
from torch import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor], defaults: dict[str, object]) -> None:
        self.params = list(params)
        self.defaults = defaults
        self.state: dict[int, dict[str, object]] = {}

    def zero_grad(self) -> None:
        pass

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0) -> None:
        defaults = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        for p in self.params:
            self.state[id(p)] = {"momentum_buffer": None}

    def step(self) -> None:
        for p in self.params:
            if id(p) not in self.state:
                self.state[id(p)] = {"momentum_buffer": None}
            state = self.state[id(p)]
            lr = float(self.defaults["lr"])
            momentum = float(self.defaults["momentum"])
            weight_decay = float(self.defaults["weight_decay"])

            dp = Tensor(0, [1], "float32")  # dummy grad — real grads unsupported
            if weight_decay != 0:
                dp = dp.add(p.mul(weight_decay))

            if momentum != 0:
                buf = state.get("momentum_buffer")
                if buf is not None:
                    buf = buf.mul(momentum).add(dp)
                else:
                    buf = dp
                state["momentum_buffer"] = buf
                dp = buf

            update = Tensor(0, [1], "float32")
            # In a real impl we'd do p -= lr * dp, but grads aren't supported


class Adam(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 0.001, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        for p in self.params:
            self.state[id(p)] = {"step": 0, "exp_avg": None, "exp_avg_sq": None}

    def step(self) -> None:
        for p in self.params:
            if id(p) not in self.state:
                self.state[id(p)] = {"step": 0, "exp_avg": None, "exp_avg_sq": None}
            state = self.state[id(p)]
            lr = float(self.defaults["lr"])
            betas = self.defaults["betas"]
            eps = float(self.defaults["eps"])
            weight_decay = float(self.defaults["weight_decay"])

            state["step"] = int(state.get("step", 0)) + 1
