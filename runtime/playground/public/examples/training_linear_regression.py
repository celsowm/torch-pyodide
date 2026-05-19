import json
import torch

# Training loop with autograd + optimizer
torch.manual_seed(42)

# Simple linear regression: y = 2*x + 1
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# Model: y = w*x + b
w = torch.randn((1, 1), requires_grad=True)
b = torch.zeros((1,), requires_grad=True)

optimizer = torch.optim.SGD([w, b], lr=0.01)
losses = []

for epoch in range(50):
    optimizer.zero_grad()
    pred = x.matmul(w) + b
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        losses.append(loss.tolist())

out = {
    "w_final": w.tolist(),
    "b_final": b.tolist(),
    "w_target": 2.0,
    "b_target": 1.0,
    "losses": losses,
    "trained": "OK" if abs(w.tolist()[0][0] - 2.0) < 0.5 else "NEED_MORE_EPOCHS"
}
print(json.dumps(out, indent=2))
