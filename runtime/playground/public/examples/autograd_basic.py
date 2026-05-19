import json
import torch

# Autograd: requires_grad + backward
a = torch.tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
b = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

# Forward pass
c = a.add(b)
d = c.mul(c)  # c^2
loss = d.sum()

# Backward pass
loss.backward()

out = {
    "loss": loss.tolist(),
    "a.grad": a.grad.tolist() if a.grad is not None else None,
    "b.grad": b.grad.tolist() if b.grad is not None else None,
    "grad_check": "OK" if a.grad is not None and b.grad is not None else "FAIL"
}
print(json.dumps(out, indent=2))
