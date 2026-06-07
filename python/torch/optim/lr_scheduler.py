"""torch.optim.lr_scheduler — minimal LR schedulers for torch-pyodide.

Implements the same public API as real PyTorch's `torch.optim.lr_scheduler`
for the small subset of schedulers that don't need extra runtime support:

* `_LRScheduler`  — base class with `step()` / `state_dict()` / `load_state_dict()`
* `StepLR`        — multiply LR by `gamma` every `step_size` epochs
* `MultiStepLR`   — multiply LR by `gamma` at the milestones
* `ExponentialLR` — multiply LR by `gamma` every epoch

These are pure-Python: no WGSL, no GPU work. They just mutate the `lr`
field of each optimizer param group, and they survive pickle round-trips
through their `state_dict` (which is a tiny dict: `last_epoch`).

Matches real PyTorch's `__init__` semantics: on construction the scheduler
performs an "initial step" (last_epoch 0, no decay yet) so the first
user-visible `step()` lands at last_epoch 1.

Usage:

    from torch.optim import SGD
    from torch.optim.lr_scheduler import StepLR

    opt = SGD(model.parameters(), lr=0.1)
    sch = StepLR(opt, step_size=30, gamma=0.1)
    for epoch in range(100):
        train_one_epoch()
        sch.step()
"""
from __future__ import annotations

from typing import List


class _LRScheduler:
    """Base class for step-based LR schedulers.

    Mirrors the contract of `torch.optim.lr_scheduler.LRScheduler`:
      * `step(epoch=None)` advances the LR for the given epoch index
      * `state_dict()` / `load_state_dict()` round-trip the epoch counter
      * `get_last_lr()` returns the current LR per param group

    Subclasses override `get_lr()` to compute the new LRs from
    `self.last_epoch` and the optimizer's `base_lrs`.

    On construction (with `last_epoch == -1`) the scheduler runs an
    "initial step" internally so that `last_epoch` ends at 0 and the
    first user-visible `step()` lands at 1. This matches the behaviour
    of recent real PyTorch versions.
    """

    def __init__(self, optimizer, last_epoch: int = -1) -> None:
        if not hasattr(optimizer, "param_groups"):
            raise TypeError(
                f"optimizer must expose a `param_groups` attribute, got {type(optimizer).__name__}"
            )

        self.optimizer = optimizer

        if last_epoch == -1:
            for group in optimizer.param_groups:
                if "initial_lr" not in group:
                    group["initial_lr"] = group["lr"]
            self.base_lrs: List[float] = [
                group["initial_lr"] for group in optimizer.param_groups
            ]
        else:
            for group in optimizer.param_groups:
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified in a param group; "
                        "the scheduler cannot resume from epoch {}".format(last_epoch)
                    )
            self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]

        self.last_epoch: int = int(last_epoch)
        # `_is_initial` mirrors real PyTorch's context manager: it tells
        # subclasses to skip the decay on the very first internal step
        # performed inside __init__.
        self._is_initial: bool = False

        if last_epoch != -1:
            # Resume mode: re-apply the formula at the loaded epoch.
            self._apply_lrs()
        else:
            # Fresh scheduler: run the "initial step" so the user-visible
            # first .step() lands at last_epoch=1. The subclass's
            # get_lr() should treat `_is_initial=True` as "no decay yet".
            self._is_initial = True
            try:
                self._apply_lrs()
            finally:
                self._is_initial = False

    def _apply_lrs(self) -> None:
        if self._is_initial:
            # First internal step: leave lr at its current value.
            self.last_epoch += 1
            return
        lrs = self.get_lr()
        for i, group in enumerate(self.optimizer.param_groups):
            group["lr"] = lrs[i]

    def get_lr(self) -> List[float]:
        raise NotImplementedError(
            f"{type(self).__name__}.get_lr must be overridden by subclasses"
        )

    def get_last_lr(self) -> List[float]:
        """Return the last computed LR per param group (read-only)."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, epoch: int | None = None) -> None:
        """Advance the scheduler.

        With no argument, increment `last_epoch` by 1 and recompute.
        Passing an explicit `epoch` jumps to that index.
        """
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = int(epoch)
        self._apply_lrs()

    def state_dict(self) -> dict[str, object]:
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.last_epoch = int(state_dict["last_epoch"])
        # Re-apply the formula so the optimizer's lr matches the loaded epoch.
        self._apply_lrs()


class StepLR(_LRScheduler):
    """Multiply LR by `gamma` every `step_size` epochs.

    Matches `torch.optim.lr_scheduler.StepLR`: on every `step()` where
    `last_epoch > 0` and `last_epoch % step_size == 0`, multiply each
    group's lr by `gamma`.
    """

    def __init__(
        self,
        optimizer,
        step_size: int = 30,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self._is_initial:
            return list(self.base_lrs)
        # `last_epoch == 0` is the "no decay yet" branch (matches real
        # PyTorch's `last_epoch == 0 or last_epoch % step_size != 0` test).
        if self.last_epoch == 0 or (self.last_epoch % self.step_size) != 0:
            # No decay this step: keep the current lr (don't reset to
            # base_lrs, which would erase previous decays).
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]


class MultiStepLR(_LRScheduler):
    """Multiply LR by `gamma` at each milestone epoch.

    Matches `torch.optim.lr_scheduler.MultiStepLR`. `milestones` is a
    sorted list of epoch indices; each one triggers a `*= gamma`.
    """

    def __init__(
        self,
        optimizer,
        milestones,  # Iterable[int]
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        milestones = sorted(int(m) for m in milestones)
        if any(m < 0 for m in milestones):
            raise ValueError("milestones must be non-negative")
        self.milestones = milestones
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self._is_initial:
            return list(self.base_lrs)
        # Apply ONE decay per step where last_epoch is a milestone. Real
        # PyTorch's MultiStepLR only decays at exact milestones, not
        # cumulatively — passing a list with duplicates is supported but
        # each duplicate counts as one decay (the docstring says
        # "If the current epoch appears in milestones n times, scale by
        # gamma to the power of n", but the canonical single-milestone
        # case is one decay per step).
        n_hits = self.milestones.count(self.last_epoch)
        if n_hits == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * (self.gamma ** n_hits) for group in self.optimizer.param_groups]


class ExponentialLR(_LRScheduler):
    """Multiply LR by `gamma` every epoch.

    Matches `torch.optim.lr_scheduler.ExponentialLR`: each `step()`
    multiplies the current lr by `gamma`.
    """

    def __init__(self, optimizer, gamma: float = 0.1, last_epoch: int = -1) -> None:
        if gamma <= 0.0 or gamma > 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self._is_initial:
            return list(self.base_lrs)
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]


class CosineAnnealingLR(_LRScheduler):
    """Cosine-anneal the LR from each base_lr down to `eta_min` over `T_max` epochs.

    Matches `torch.optim.lr_scheduler.CosineAnnealingLR`. The formula at
    epoch `last_epoch` is::

        lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * last_epoch / T_max))

    `T_max` must be positive. The very first call (last_epoch=0 from
    `__init__`'s initial step) gives lr = base_lr.
    """

    def __init__(
        self,
        optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if T_max <= 0:
            raise ValueError(f"T_max must be positive, got {T_max}")
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self._is_initial:
            return list(self.base_lrs)
        import math
        # Real PyTorch computes the new lr at last_epoch in [0, T_max] and
        # clamps the cosine factor to 0 when last_epoch >= T_max.
        e = min(self.last_epoch, self.T_max)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * e / self.T_max))
        return [
            self.eta_min + (base_lr - self.eta_min) * cos_factor
            for base_lr in self.base_lrs
        ]


class ReduceLROnPlateau:
    """Reduce LR when a metric has stopped improving.

    Matches `torch.optim.lr_scheduler.ReduceLROnPlateau`. Unlike the other
    schedulers, this one takes a `step(metrics)` call (you pass the
    monitored metric value) and is not a subclass of `_LRScheduler`.

    Args:
        optimizer: wrapped optimizer.
        mode: 'min' (lower is better) or 'max' (higher is better).
        factor: factor by which the LR is reduced (lr *= factor).
        patience: number of steps with no improvement after which LR is reduced.
        threshold: threshold for measuring the new optimum.
        threshold_mode: 'rel' (relative) or 'abs' (absolute).
        cooldown: number of steps to wait before resuming normal operation.
        min_lr: lower bound on the LR per param group (scalar or list of scalars).
        eps: minimal decay applied to the LR.
    """

    def __init__(
        self,
        optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float | list[float] = 0.0,
        eps: float = 1e-8,
    ) -> None:
        if factor >= 1.0:
            raise ValueError(f"factor must be < 1.0, got {factor}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        if threshold_mode not in ("rel", "abs"):
            raise ValueError(f"threshold_mode must be 'rel' or 'abs', got {threshold_mode}")
        if not hasattr(optimizer, "param_groups"):
            raise TypeError("optimizer must expose a `param_groups` attribute")

        self.optimizer = optimizer
        self.mode = mode
        self.factor = float(factor)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.threshold_mode = threshold_mode
        self.cooldown = int(cooldown)
        self.cooldown_counter = 0
        self.eps = float(eps)

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected min_lr per param group")
            self.min_lrs = [float(v) for v in min_lr]
        else:
            self.min_lrs = [float(min_lr)] * len(optimizer.param_groups)

        for group in optimizer.param_groups:
            if "initial_lr" not in group:
                group["initial_lr"] = group["lr"]
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]

        self.best: float = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs: int = 0
        self.last_epoch: int = 0

    def is_better(self, a: float, best: float) -> bool:
        if self.mode == "min" and self.threshold_mode == "rel":
            return a < best * (1.0 - self.threshold)
        if self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold
        if self.mode == "max" and self.threshold_mode == "rel":
            return a > best * (1.0 + self.threshold)
        # mode == "max", threshold_mode == "abs"
        return a > best + self.threshold

    def step(self, metrics: float, epoch: int | None = None) -> None:
        """Advance the scheduler by one step using the current metric value."""
        current = float(metrics)
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = int(epoch)

    def _reduce_lr(self) -> None:
        for i, group in enumerate(self.optimizer.param_groups):
            old_lr = float(group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                group["lr"] = new_lr

    def state_dict(self) -> dict[str, object]:
        return {
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "cooldown_counter": self.cooldown_counter,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.best = float(state_dict["best"])
        self.num_bad_epochs = int(state_dict["num_bad_epochs"])
        self.cooldown_counter = int(state_dict["cooldown_counter"])
        self.last_epoch = int(state_dict["last_epoch"])
