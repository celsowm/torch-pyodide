import json

import torch

torch.manual_seed(42)
a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
b = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

c = torch.cat([a, b])
e = a.expand([3, 3])

loss = c.sum() + e.sum()
loss.backward()

cond = torch.tensor([True, False, True])
x = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
w = torch.where(cond, x, y)

x.grad = None
y.grad = None
w.sum().backward()

print(json.dumps({
    "cat": c.tolist(),
    "expand": e.tolist(),
    "where": w.tolist(),
    "grad_a": [round(v, 4) for v in a.grad.tolist()],
    "grad_b": [round(v, 4) for v in b.grad.tolist()],
    "grad_x": [round(v, 4) for v in x.grad.tolist()],
    "grad_y": [round(v, 4) for v in y.grad.tolist()],
}, sort_keys=True))
