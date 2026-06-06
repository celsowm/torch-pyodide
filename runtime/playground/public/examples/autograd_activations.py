import json

import torch
import torch.nn.functional as F

torch.manual_seed(42)
x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], requires_grad=True)

grads = {}
for name, fn in [
    ("sigmoid", lambda: F.sigmoid(x)),
    ("tanh", lambda: F.tanh(x)),
    ("gelu", lambda: F.gelu(x)),
    ("silu", lambda: F.silu(x)),
    ("leaky_relu", lambda: F.leaky_relu(x, 0.1)),
    ("softmax", lambda: F.softmax(x, dim=-1)),
    ("relu", lambda: F.relu(x)),
    ("elu", lambda: F.elu(x, 1.0)),
    ("celu", lambda: F.celu(x, 1.0)),
]:
    y = fn()
    loss = y.sum()
    x.grad = None
    loss.backward()
    grads[name] = [round(v, 4) for v in x.grad.tolist()]

print(json.dumps(grads, sort_keys=True))
