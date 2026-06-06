import json

import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 0.5, -1.0]], requires_grad=True)
target = torch.tensor([0])

loss = F.cross_entropy(logits, target)
loss.backward()

print(json.dumps({
    "loss": round(float(loss.detach()), 5),
    "grad": [round(v, 4) for v in logits.grad.tolist()[0]],
}, sort_keys=True))
