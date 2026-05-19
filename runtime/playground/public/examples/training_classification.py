import json
import torch

# Training loop: Adam optimizer on simple classification
torch.manual_seed(42)

# Input features (2D points), Labels (0 or 1)
x = torch.tensor([
    [1.0, 1.0],
    [2.0, 2.0],
    [3.0, 3.0],
    [-1.0, -1.0],
    [-2.0, -2.0],
    [-3.0, -3.0],
])
y = torch.tensor([0, 0, 0, 1, 1, 1])

# Simple linear model
w = torch.randn((2, 1), requires_grad=True)
b = torch.zeros((1,), requires_grad=True)

optimizer = torch.optim.Adam([w, b], lr=0.1)
losses = []

for epoch in range(100):
    optimizer.zero_grad()
    logits = x.matmul(w) + b
    loss = torch.nn.functional.cross_entropy(
        torch.cat([-logits, logits], dim=1), y
    )
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        losses.append(loss.tolist())

# Predictions
with torch.no_grad():
    logits = x.matmul(w) + b
    preds = torch.where(logits.squeeze() > 0, 1, 0).tolist()

out = {
    "w_final": w.tolist(),
    "b_final": b.tolist(),
    "losses": losses,
    "predictions": preds,
    "accuracy": sum(1 for p, t in zip(preds, y.tolist()) if p == t) / len(y),
    "status": "OK"
}
print(json.dumps(out, indent=2))
