"""
BatchNorm2d in training mode: forward + running stats + backward + end-to-end training.

Exercises Fase 12.5 BN-training path. Emits a JSON object with:
- Forward output (training mode) and the expected closed-form value
- Running stats after 1 step
- num_batches_tracked after 3 steps
- Gradients via autograd vs. finite differences (correctness check)
- Eval-mode output (uses running stats, not batch stats)
- Loss decreased after a few optimizer steps (end-to-end training works)
"""
import json
import math
import torch
import torch.nn as nn


def main() -> None:
    # ── 1. Forward in training mode ─────────────────────────────────
    torch.manual_seed(0)
    bn = nn.BatchNorm1d(4)
    bn.weight = nn.Parameter(torch.ones((4,)))
    bn.bias = nn.Parameter(torch.zeros((4,)))
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0]],
        requires_grad=True,
    )

    bn.train()
    y = bn(x)
    # Expected closed-form: (x - mean) / sqrt(var + eps), per channel.
    mean = x.mean(dim=0)
    var = ((x - mean) ** 2).mean(dim=0)
    expected = (x - mean) * (var + bn.eps).rsqrt()
    diff = (y - expected).abs().max()
    first_forward_close = bool(diff.item() < 1e-3)

    # ── 2. Running stats after 1 step ───────────────────────────────
    running_mean_1 = [round(v, 6) for v in bn.running_mean.tolist()]
    running_var_1 = [round(v, 6) for v in bn.running_var.tolist()]

    # ── 3. num_batches_tracked increments on each forward ────────────
    nbt_before_raw = bn.num_batches_tracked.tolist()
    nbt_before = nbt_before_raw[0] if isinstance(nbt_before_raw, list) else nbt_before_raw
    for _ in range(2):
        bn(x)
    nbt_after_raw = bn.num_batches_tracked.tolist()
    nbt_after = nbt_after_raw[0] if isinstance(nbt_after_raw, list) else nbt_after_raw

    # ── 4. Gradients vs. finite differences ──────────────────────────
    # Recompute on a fresh input. We use loss = (y * y).sum() so the
    # upstream gradient is 2*y (non-constant) — without this, dl/dx_hat is
    # constant and the standard BN backward gives 0 gradient for x by
    # symmetry (the centered normalized values cancel out).
    x_g = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0]],
        requires_grad=True,
    )
    w_g = nn.Parameter(torch.ones((4,)))
    b_g = nn.Parameter(torch.zeros((4,)))
    bn2 = nn.BatchNorm1d(4)
    bn2.weight = w_g
    bn2.bias = b_g
    bn2.train()
    y_g = bn2(x_g)
    loss = (y_g * y_g).sum()
    loss.backward()
    x_grad = x_g.grad.tolist()
    w_grad = w_g.grad.tolist()
    b_grad = b_g.grad.tolist()

    # Finite differences for x
    eps_fd = 1e-3
    x_grad_fd = [[0.0] * 4 for _ in range(3)]
    x_data = x_g.tolist()  # work in pure-Python to avoid in-place Tensor issues
    for r in range(3):
        for c in range(4):
            x_plus_data = [row[:] for row in x_data]
            x_plus_data[r][c] = x_plus_data[r][c] + eps_fd
            x_minus_data = [row[:] for row in x_data]
            x_minus_data[r][c] = x_minus_data[r][c] - eps_fd
            x_plus = torch.tensor(x_plus_data, requires_grad=False)
            x_minus = torch.tensor(x_minus_data, requires_grad=False)
            bn2.zero_grad()
            y_p = bn2(x_plus)
            y_m = bn2(x_minus)
            x_grad_fd[r][c] = ((y_p * y_p).sum().item() - (y_m * y_m).sum().item()) / (2 * eps_fd)

    # Finite differences for weight
    w_grad_fd = [0.0] * 4
    w_data = w_g.tolist()
    b_data = b_g.tolist()
    for c in range(4):
        w_plus_data = w_data[:]
        w_plus_data[c] = w_plus_data[c] + eps_fd
        w_minus_data = w_data[:]
        w_minus_data[c] = w_minus_data[c] - eps_fd
        bn3 = nn.BatchNorm1d(4)
        bn3.weight = nn.Parameter(torch.tensor(w_plus_data))
        bn3.bias = nn.Parameter(torch.tensor(b_data))
        bn3.train()
        y_p = bn3(torch.tensor(x_data))
        bn4 = nn.BatchNorm1d(4)
        bn4.weight = nn.Parameter(torch.tensor(w_minus_data))
        bn4.bias = nn.Parameter(torch.tensor(b_data))
        bn4.train()
        y_m = bn4(torch.tensor(x_data))
        w_grad_fd[c] = ((y_p * y_p).sum().item() - (y_m * y_m).sum().item()) / (2 * eps_fd)

    # Finite differences for bias
    b_grad_fd = [0.0] * 4
    for c in range(4):
        b_plus_data = b_data[:]
        b_plus_data[c] = b_plus_data[c] + eps_fd
        b_minus_data = b_data[:]
        b_minus_data[c] = b_minus_data[c] - eps_fd
        bn5 = nn.BatchNorm1d(4)
        bn5.weight = nn.Parameter(torch.tensor(w_data))
        bn5.bias = nn.Parameter(torch.tensor(b_plus_data))
        bn5.train()
        y_p = bn5(torch.tensor(x_data))
        bn6 = nn.BatchNorm1d(4)
        bn6.weight = nn.Parameter(torch.tensor(w_data))
        bn6.bias = nn.Parameter(torch.tensor(b_minus_data))
        bn6.train()
        y_m = bn6(torch.tensor(x_data))
        b_grad_fd[c] = ((y_p * y_p).sum().item() - (y_m * y_m).sum().item()) / (2 * eps_fd)

    def max_abs_diff(a, b):
        return max(abs(ai - bi) for ai, bi in zip(a, b))

    # For x_grad_fd compare with x_grad flattened
    x_grad_flat = [v for row in x_grad for v in row]
    x_grad_fd_flat = [v for row in x_grad_fd for v in row]
    grad_x_max_abs_diff = max_abs_diff(x_grad_flat, x_grad_fd_flat)
    grad_w_max_abs_diff = max_abs_diff(w_grad, w_grad_fd)
    grad_b_max_abs_diff = max_abs_diff(b_grad, b_grad_fd)

    # ── 5. Eval mode uses running stats ──────────────────────────────
    bn.eval()
    x_eval = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0]],
    )
    y_eval = bn(x_eval)
    # In eval mode: y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
    expected_eval = (
        (x_eval - torch.tensor(bn.running_mean.tolist()))
        * (torch.tensor(bn.running_var.tolist()) + bn.eps).rsqrt()
    )
    eval_first_y_uses_running_stats = bool(
        (y_eval - expected_eval).abs().max().item() < 1e-3
    )

    # ── 6. End-to-end tiny training loop ─────────────────────────────
    # Train a Linear -> BN -> Linear -> MSE model to verify that the
    # BN-training path is actually usable for a real model.
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.BatchNorm1d(8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    target = torch.tensor([[0.5], [1.0], [1.5]])
    losses = []
    for step in range(20):
        opt.zero_grad()
        model.train()
        out = model(x.detach())
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    loss_decreased = losses[-1] < losses[0]

    out = {
        # Forward
        "forward_shape": list(y.shape),
        "first_forward_y": [round(v, 4) for v in y[0].tolist()],
        "expected_first_y": [round(v, 4) for v in expected[0].tolist()],
        "first_forward_close_to_expected": first_forward_close,
        # Running stats
        "running_mean_after_1_step": running_mean_1,
        "running_var_after_1_step": running_var_1,
        "num_batches_tracked_before": nbt_before,
        "num_batches_tracked_after_2_more_steps": nbt_after,
        # Gradients (autograd)
        "x_grad_first": [round(v, 6) for v in x_grad[0]],
        "w_grad_first": [round(v, 6) for v in w_grad],
        "b_grad_first": [round(v, 6) for v in b_grad],
        # Gradients (finite differences)
        "x_grad_finite_diff": [round(v, 6) for v in x_grad_fd_flat],
        "w_grad_finite_diff": [round(v, 6) for v in w_grad_fd],
        "b_grad_finite_diff": [round(v, 6) for v in b_grad_fd],
        "grad_x_max_abs_diff": float(grad_x_max_abs_diff),
        "grad_w_max_abs_diff": float(grad_w_max_abs_diff),
        "grad_b_max_abs_diff": float(grad_b_max_abs_diff),
        # Eval mode
        "eval_first_y": [round(v, 4) for v in y_eval[0].tolist()],
        "eval_first_y_uses_running_stats": eval_first_y_uses_running_stats,
        # Training loop
        "losses_first_5": [round(v, 6) for v in losses[:5]],
        "losses_last_5": [round(v, 6) for v in losses[-5:]],
        "loss_decreased": loss_decreased,
    }
    print(json.dumps(out, indent=2))


main()
