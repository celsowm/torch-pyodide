"""RAdam optimizer parity smoke."""
import json

import torch


# Deterministic initial weights AND data: both runtimes start from exactly
# the same state so the optimizer trajectories are directly comparable.
# We use raw leaf tensors with requires_grad=True (not nn.Linear) so the
# optimizer state is unambiguous across runtimes.
w = torch.tensor([[0.1], [0.2], [0.3]], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

optimizer = torch.optim.RAdam([w, b], lr=0.01)

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

losses = []
for step in range(50):
    pred = x.matmul(w) + b
    loss = ((pred - y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.item()))


print(json.dumps({
    "loss_start": round(losses[0], 4),
    "loss_end": round(losses[-1], 4),
    "loss_mid": round(losses[25], 4),
    "weight": [round(float(v), 4) for row in w.tolist() for v in row],
    "bias": [round(float(v), 4) for v in b.tolist()],
    "final_loss": round(losses[-1], 4),
}, sort_keys=True))
