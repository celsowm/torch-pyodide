import json
import torch

# Select with autograd
torch.manual_seed(42)

x = torch.randn((3, 4), requires_grad=True)
row = x.select(0, 1)

loss = (row ** 2).sum()
loss.backward()

out = {
    "x_shape": list(x.shape),
    "row_shape": list(row.shape),
    "loss": loss.tolist(),
    "x.grad_nonzero": bool((x.grad != 0).any()) if x.grad is not None else False,
    "x.grad_shape": list(x.grad.shape) if x.grad is not None else None,
}
print(json.dumps(out, indent=2))
