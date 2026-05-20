import json
import torch

# max / argmax with autograd
torch.manual_seed(42)

x = torch.randn((2, 5), requires_grad=True)
y = x.max()

y.backward()

out = {
    "x_shape": list(x.shape),
    "max_val": y.tolist(),
    "x.grad_nonzero": bool((x.grad != 0).any()) if x.grad is not None else False,
    "x.grad_shape": list(x.grad.shape) if x.grad is not None else None,
}
print(json.dumps(out, indent=2))
