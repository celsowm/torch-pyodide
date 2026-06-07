"""Tests for torch.optim.lr_scheduler (StepLR / MultiStepLR / ExponentialLR).

These are pure-Python classes; they don't touch the runtime. We test
them against a fake optimizer (just a list of param-groups) to avoid
spinning up the WebGPU runtime inside the pytest process. The
end-to-end version of the scheduler is exercised in the browser by
the `real_model_pretrained_vgg` test, which actually wires the
scheduler up to a real `torch.optim.Adam`.

The schedulers match real PyTorch's `_initial_step` semantics: on
construction the scheduler runs an internal step that lands at
`last_epoch=0` (no decay yet) so the first user-visible `step()`
lands at `last_epoch=1`. This means the test loop counts user-visible
steps, NOT internal calls to `step()`.
"""
from __future__ import annotations

import math
from typing import List


class _FakeParamGroup:
    def __init__(self, lr: float) -> None:
        self.lr = lr


class _FakeOptimizer:
    """Stand-in for `torch.optim.Optimizer` that exposes `param_groups`.

    The LR schedulers only need to read `group["lr"]` and write it back,
    plus read `group["initial_lr"]`. This fake supports both.
    """

    def __init__(self, lrs: List[float]) -> None:
        self.param_groups: list[dict[str, float]] = []
        for lr in lrs:
            group: dict[str, float] = {"lr": lr}
            self.param_groups.append(group)


def test_steplr_decays_at_each_step_size_boundary() -> None:
    opt = _FakeOptimizer([1.0])
    from torch.optim.lr_scheduler import StepLR

    sch = StepLR(opt, step_size=3, gamma=0.5)
    # After construction, the internal _initial_step puts last_epoch=0
    # without applying decay. So lr is still 1.0.
    assert math.isclose(opt.param_groups[0]["lr"], 1.0, rel_tol=0, abs_tol=1e-12)
    # user step 1: last_epoch 0->1, 1 % 3 != 0, no decay.
    sch.step()
    assert math.isclose(opt.param_groups[0]["lr"], 1.0, rel_tol=0, abs_tol=1e-12)
    # user step 2: last_epoch 1->2, 2 % 3 != 0, no decay.
    sch.step()
    assert math.isclose(opt.param_groups[0]["lr"], 1.0, rel_tol=0, abs_tol=1e-12)
    # user step 3: last_epoch 2->3, 3 % 3 == 0 (and != 0), decay.
    sch.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.5, rel_tol=0, abs_tol=1e-12)
    # user step 6: last_epoch 5->6, 6 % 3 == 0, second decay -> 0.25.
    for _ in range(3):
        sch.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.25, rel_tol=0, abs_tol=1e-12)


def test_steplr_with_step_size_one_decays_every_user_step() -> None:
    opt = _FakeOptimizer([0.1])
    from torch.optim.lr_scheduler import StepLR

    sch = StepLR(opt, step_size=1, gamma=0.1)
    # user step 1: last_epoch 0->1, 1 % 1 == 0, decay.
    sch.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.01, rel_tol=0, abs_tol=1e-12)
    # user step 2: last_epoch 1->2, 2 % 1 == 0, second decay -> 0.001.
    sch.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.001, rel_tol=0, abs_tol=1e-12)


def test_steplr_get_last_lr_returns_current_value() -> None:
    opt = _FakeOptimizer([0.1])
    from torch.optim.lr_scheduler import StepLR

    sch = StepLR(opt, step_size=1, gamma=0.1)
    sch.step()
    sch.step()
    assert math.isclose(sch.get_last_lr()[0], 0.001, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(opt.param_groups[0]["lr"], 0.001, rel_tol=0, abs_tol=1e-12)


def test_steplr_state_dict_roundtrip() -> None:
    opt = _FakeOptimizer([0.5])
    from torch.optim.lr_scheduler import StepLR

    sch = StepLR(opt, step_size=4, gamma=0.25)
    for _ in range(7):
        sch.step()
    snapshot = sch.state_dict()
    expected_lr = opt.param_groups[0]["lr"]

    sch2 = StepLR(opt, step_size=4, gamma=0.25)
    sch2.step()  # one step forward
    sch2.load_state_dict(snapshot)
    assert sch2.last_epoch == snapshot["last_epoch"]
    assert math.isclose(opt.param_groups[0]["lr"], expected_lr, rel_tol=0, abs_tol=1e-12)


def test_steplr_explicit_epoch_jump() -> None:
    opt = _FakeOptimizer([1.0])
    from torch.optim.lr_scheduler import StepLR

    sch = StepLR(opt, step_size=2, gamma=0.5)
    # epoch=4 -> last_epoch=4 -> 4 % 2 == 0 (and != 0) -> one decay.
    # StepLR applies at most one decay per step (regardless of how many
    # step_size boundaries were crossed).
    sch.step(epoch=4)
    assert math.isclose(opt.param_groups[0]["lr"], 0.5, rel_tol=0, abs_tol=1e-12)


def test_steplr_rejects_invalid_gamma() -> None:
    from torch.optim.lr_scheduler import StepLR

    opt = _FakeOptimizer([1.0])
    for bad in (0.0, -0.1):
        try:
            StepLR(opt, step_size=1, gamma=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"StepLR(gamma={bad}) should have raised")


def test_steplr_rejects_non_positive_step_size() -> None:
    from torch.optim.lr_scheduler import StepLR

    opt = _FakeOptimizer([1.0])
    for bad in (0, -3):
        try:
            StepLR(opt, step_size=bad, gamma=0.5)
        except ValueError:
            pass
        else:
            raise AssertionError(f"StepLR(step_size={bad}) should have raised")


def test_multisteplr_hits_each_milestone_once() -> None:
    opt = _FakeOptimizer([1.0])
    from torch.optim.lr_scheduler import MultiStepLR

    sch = MultiStepLR(opt, milestones=[2, 5, 8], gamma=0.5)
    # After construction: lr unchanged (1.0).
    assert math.isclose(opt.param_groups[0]["lr"], 1.0, rel_tol=0, abs_tol=1e-12)
    # Advance to last_epoch=2: 1 milestone hit, lr *= gamma -> 0.5.
    sch.step(epoch=2)
    assert math.isclose(opt.param_groups[0]["lr"], 0.5, rel_tol=0, abs_tol=1e-12)
    # Advance to last_epoch=5: 1 more hit, lr *= gamma -> 0.25.
    sch.step(epoch=5)
    assert math.isclose(opt.param_groups[0]["lr"], 0.25, rel_tol=0, abs_tol=1e-12)
    # Advance to last_epoch=8: 1 more hit, lr *= gamma -> 0.125.
    sch.step(epoch=8)
    assert math.isclose(opt.param_groups[0]["lr"], 0.125, rel_tol=0, abs_tol=1e-12)


def test_exponentiallr_decays_every_user_step() -> None:
    opt = _FakeOptimizer([1.0])
    from torch.optim.lr_scheduler import ExponentialLR

    sch = ExponentialLR(opt, gamma=0.9)
    for user_step in range(1, 5):
        sch.step()
        assert math.isclose(
            opt.param_groups[0]["lr"],
            0.9 ** user_step,
            rel_tol=0,
            abs_tol=1e-12,
        )


def test_steplr_multiple_param_groups_independent_lrs() -> None:
    opt = _FakeOptimizer([0.1, 1.0])
    from torch.optim.lr_scheduler import StepLR

    sch = StepLR(opt, step_size=1, gamma=0.1)
    sch.step()  # user step 1: last_epoch=1, both groups decay.
    assert math.isclose(opt.param_groups[0]["lr"], 0.01, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(opt.param_groups[1]["lr"], 0.1, rel_tol=0, abs_tol=1e-12)


def test_steplr_records_initial_lr_on_first_use() -> None:
    # Match real PyTorch: after StepLR is constructed, the param group
    # carries an `initial_lr` so subsequent restarts are stable.
    opt = _FakeOptimizer([0.05])
    from torch.optim.lr_scheduler import StepLR

    StepLR(opt, step_size=1, gamma=0.1)
    assert math.isclose(opt.param_groups[0]["initial_lr"], 0.05, rel_tol=0, abs_tol=1e-12)
