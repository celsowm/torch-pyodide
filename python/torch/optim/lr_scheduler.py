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
