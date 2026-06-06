"""Optimizer state_dict save/load roundtrip.

Trains an Adam optimizer for 5 steps, snapshots its state_dict, rebuilds the
optimizer with the same hyperparameters, loads the snapshot, and verifies that
the new optimizer continues training without raising.

The same script runs in real PyTorch and in torch-pyodide; the browser test
compares the JSON payload against the real-PyTorch reference.
"""
import base64
import io
import json

import torch


# Deterministic initial weights AND data so both runtimes start from exactly
# the same state. We use raw leaf tensors with requires_grad=True (not
# nn.Linear) so the optimizer state is unambiguous across runtimes.
w = torch.tensor([[0.1], [0.2], [0.3]], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

x = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [1.0, 1.1, 1.2],
    [0.5, 0.0, -0.5],
    [-0.3, 0.7, 0.2],
    [0.8, -0.4, 0.1],
    [0.2, 0.9, -0.1],
])
y = torch.tensor([[0.5], [1.5], [2.5], [3.5], [0.0], [0.7], [0.6], [1.0]])


def _train_n_steps(opt: torch.optim.Optimizer, params: list, n: int) -> list[float]:
    losses: list[float] = []
    for _ in range(n):
        pred = x.matmul(params[0]) + params[1]
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(round(float(loss.item()), 4))
    return losses


# ── 1. Train first half, snapshot the optimizer state_dict ────────────────
opt_a = torch.optim.Adam([w, b], lr=0.01)
losses_a_first = _train_n_steps(opt_a, [w, b], 5)
opt_state_bytes = io.BytesIO()
torch.save(opt_a.state_dict(), opt_state_bytes)
opt_state_bytes.seek(0)
state_b64 = base64.b64encode(opt_state_bytes.getvalue()).decode("ascii")

# ── 2. Rebuild optimizer, restore state, continue training ────────────────
opt_b = torch.optim.Adam([w, b], lr=0.01)
loaded = torch.load(io.BytesIO(base64.b64decode(state_b64)))
opt_b.load_state_dict(loaded)
losses_b_second = _train_n_steps(opt_b, [w, b], 5)

# ── 3. Fresh optimizer (no state restoration) for comparison ──────────────
w_fresh = torch.tensor([[0.1], [0.2], [0.3]], requires_grad=True)
b_fresh = torch.tensor([0.0], requires_grad=True)
opt_fresh = torch.optim.Adam([w_fresh, b_fresh], lr=0.01)
_train_n_steps(opt_fresh, [w_fresh, b_fresh], 5)
opt_fresh_second = _train_n_steps(opt_fresh, [w_fresh, b_fresh], 5)

print(json.dumps({
    "losses_a_first": losses_a_first,
    "losses_b_second": losses_b_second,
    "losses_fresh_second": opt_fresh_second,
    "state_b64_length": len(state_b64),
}, sort_keys=True))
