import json

import torch
import torch.nn as nn

torch.manual_seed(42)
w = torch.tensor([0.0], requires_grad=True)
target = torch.tensor([2.0])
optimizer = torch.optim.SGD([w], lr=0.1)

loss = (w - target).pow(2).mean()
loss.backward()

optimizer.step()
optimizer.zero_grad()

print(json.dumps({
    "loss": round(float(loss.detach()), 4),
    "updated_w": round(float(w.detach()), 4),
}, sort_keys=True))
