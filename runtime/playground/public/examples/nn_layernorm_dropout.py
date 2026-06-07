"""
LayerNorm + Dropout: forward + autograd backward + end-to-end training.

Exercises Fase 12.6 — LayerNorm normalization with affine and Dropout with mask-based
backward. Emits a JSON object with:
- Forward output (training mode) and the expected closed-form value
- Eval-mode output (Dropout is identity, LN uses same params)
- Gradients via autograd vs. finite differences (correctness check)
- Loss decreased after a few optimizer steps (end-to-end training works)
"""
import json
import math
import torch
import torch.nn as nn


def main() -> None:
    # ── 1. Forward in training mode ─────────────────────────────────
    torch.manual_seed(0)
    ln = nn.LayerNorm(4)
    ln.weight = nn.Parameter(torch.tensor([1.5, 0.5, 2.0, 1.0]))
    ln.bias = nn.Parameter(torch.tensor([0.1, -0.2, 0.3, 0.0]))
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0]],
        requires_grad=True,
    )
    y = ln(x)
    forward_shape = list(y.shape)

    # Closed-form expected output (LN normalizes over the last dim per row).
    # row means: 2.5, 6.5, 10.5; row var (biased): 1.25, 1.25, 1.25; std=sqrt(1.25)
    std = math.sqrt(1.25)
    inv_std = 1.0 / std
    x_centered = [
        [-1.5, -0.5, 0.5, 1.5],
        [-1.5, -0.5, 0.5, 1.5],
        [-1.5, -0.5, 0.5, 1.5],
    ]
    x_hat = [[c * inv_std for c in row] for row in x_centered]
    w = [1.5, 0.5, 2.0, 1.0]
    b = [0.1, -0.2, 0.3, 0.0]
    expected_y = [[x_hat[i][j] * w[j] + b[j] for j in range(4)] for i in range(3)]
    actual_y = y.tolist()
    max_diff = max(
        abs(actual_y[i][j] - expected_y[i][j]) for i in range(3) for j in range(4)
    )
    forward_close = max_diff < 1e-3

    # ── 2. Eval mode sanity (no Dropout yet, just LN) ───────────────
    ln.eval()
    y_eval = ln(x)
    # In eval mode, LN uses the same parameters; the *forward output* is identical
    # because the only state is the affine parameters, not running stats.
    eval_close_training = max(
        abs(y_eval.tolist()[i][j] - y.tolist()[i][j]) for i in range(3) for j in range(4)
    ) < 1e-4
    ln.train()

    # ── 3. Autograd vs finite differences for LN ───────────────────
    torch.manual_seed(1)
    ln2 = nn.LayerNorm(4)
    ln2.weight = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]))
    ln2.bias = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0]))
    x_g = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0]],
        requires_grad=True,
    )
    w_g = ln2.weight
    b_g = ln2.bias
    y_g = ln2(x_g)
    # Use a non-constant upstream gradient to get a non-zero x gradient.
    loss_g = (y_g * y_g).sum()
    loss_g.backward()
    x_grad_auto = x_g.grad.tolist()
    w_grad_auto = w_g.grad.tolist()
    b_grad_auto = b_g.grad.tolist()

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
            ln2.zero_grad()
            y_p = ln2(x_plus)
            y_m = ln2(x_minus)
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
        ln3 = nn.LayerNorm(4)
        ln3.weight = nn.Parameter(torch.tensor(w_plus_data))
        ln3.bias = nn.Parameter(torch.tensor(b_data))
        y_p = ln3(torch.tensor(x_data))
        ln4 = nn.LayerNorm(4)
        ln4.weight = nn.Parameter(torch.tensor(w_minus_data))
        ln4.bias = nn.Parameter(torch.tensor(b_data))
        y_m = ln4(torch.tensor(x_data))
        w_grad_fd[c] = ((y_p * y_p).sum().item() - (y_m * y_m).sum().item()) / (2 * eps_fd)

    # Finite differences for bias
    b_grad_fd = [0.0] * 4
    for c in range(4):
        b_plus_data = b_data[:]
        b_plus_data[c] = b_plus_data[c] + eps_fd
        b_minus_data = b_data[:]
        b_minus_data[c] = b_minus_data[c] - eps_fd
        ln5 = nn.LayerNorm(4)
        ln5.weight = nn.Parameter(torch.tensor(w_data))
        ln5.bias = nn.Parameter(torch.tensor(b_plus_data))
        y_p = ln5(torch.tensor(x_data))
        ln6 = nn.LayerNorm(4)
        ln6.weight = nn.Parameter(torch.tensor(w_data))
        ln6.bias = nn.Parameter(torch.tensor(b_minus_data))
        y_m = ln6(torch.tensor(x_data))
        b_grad_fd[c] = ((y_p * y_p).sum().item() - (y_m * y_m).sum().item()) / (2 * eps_fd)

    grad_x_max_abs_diff = max(
        abs(x_grad_auto[i][j] - x_grad_fd[i][j]) for i in range(3) for j in range(4)
    )
    grad_w_max_abs_diff = max(abs(w_grad_auto[j] - w_grad_fd[j]) for j in range(4))
    grad_b_max_abs_diff = max(abs(b_grad_auto[j] - b_grad_fd[j]) for j in range(4))

    # ── 4. Dropout training vs eval ─────────────────────────────────
    torch.manual_seed(2)
    drop = nn.Dropout(p=0.5)
    drop.train()
    x_d = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    y_d_train = drop(x_d)
    # Save mask: in our implementation, y_d_train = x * mask * 2
    # where mask is 0 or 1. Count zero entries.
    train_zero_frac = sum(1 for v in y_d_train.tolist()[0] if v == 0.0) / 4.0
    drop.eval()
    y_d_eval = drop(x_d)
    eval_zero_frac = sum(1 for v in y_d_eval.tolist()[0] if v == 0.0) / 4.0

    # ── 5. Dropout gradient sanity ──────────────────────────────────
    torch.manual_seed(3)
    x_dg = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]], requires_grad=True
    )
    drop2 = nn.Dropout(p=0.0)  # p=0 is identity forward; use it for a clean grad check
    drop2.train()
    y_dg = drop2(x_dg)
    y_dg.sum().backward()
    dropout_grad_at_p0 = x_dg.grad.tolist()  # should be all 1.0s

    # ── 6. End-to-end training: Linear -> LayerNorm -> Dropout -> Linear
    torch.manual_seed(4)
    in_dim, hidden_dim, out_dim = 8, 16, 4
    net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(0.5),
        nn.Linear(hidden_dim, out_dim),
    )

    # Fake regression dataset: y = sin(x[:, 0]) + 0.5 * x[:, 1]
    n_samples = 32
    x_train = torch.randn((n_samples, in_dim))
    y_train = torch.sin(x_train[:, 0]) + 0.5 * x_train[:, 1]

    opt = torch.optim.SGD(net.parameters(), lr=0.05)

    def forward_loss() -> float:
        net.train()
        opt.zero_grad()
        pred = net(x_train)
        loss = ((pred - y_train.unsqueeze(1)) ** 2).mean()
        loss.backward()
        opt.step()
        return float(loss.item())

    losses = [forward_loss() for _ in range(40)]
    loss_decreased = losses[-1] < losses[0] * 0.5  # at least 50% reduction

    # ── 7. Eval mode of trained network ─────────────────────────────
    net.eval()
    with torch.no_grad():
        y_pred_eval = net(x_train)
    eval_loss = float(((y_pred_eval - y_train.unsqueeze(1)) ** 2).mean().item())

    out = {
        "forward_shape": forward_shape,
        "first_forward_y": [round(v, 4) for v in y.tolist()[0]],
        "expected_first_y": [round(v, 4) for v in expected_y[0]],
        "first_forward_close_to_expected": bool(forward_close),
        "eval_first_y_equals_training_first_y": bool(eval_close_training),
        "x_grad_first": [round(v, 4) for v in x_grad_auto[0]],
        "w_grad_first": [round(v, 4) for v in w_grad_auto],
        "b_grad_first": [round(v, 4) for v in b_grad_auto],
        "x_grad_finite_diff": [round(v, 4) for v in x_grad_fd[0]],
        "w_grad_finite_diff": [round(v, 4) for v in w_grad_fd],
        "b_grad_finite_diff": [round(v, 4) for v in b_grad_fd],
        "grad_x_max_abs_diff": float(grad_x_max_abs_diff),
        "grad_w_max_abs_diff": float(grad_w_max_abs_diff),
        "grad_b_max_abs_diff": float(grad_b_max_abs_diff),
        "dropout_train_zero_frac": float(train_zero_frac),
        "dropout_eval_zero_frac": float(eval_zero_frac),
        "dropout_grad_at_p0": dropout_grad_at_p0,
        "losses_first_5": [round(v, 6) for v in losses[:5]],
        "losses_last_5": [round(v, 6) for v in losses[-5:]],
        "loss_decreased": bool(loss_decreased),
        "eval_loss": float(eval_loss),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
