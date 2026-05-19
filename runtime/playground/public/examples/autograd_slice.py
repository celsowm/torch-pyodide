import json
import torch

# Slice with autograd (runtime-backed backprop)
torch.manual_seed(42)

x = torch.randn((4, 6), requires_grad=True)

# Slice the tensor
sliced = x[1:3, 2:5]

# Compute a loss from the sliced portion
loss = (sliced ** 2).sum()

# Backward: gradients flow back to the original tensor
loss.backward()

out = {
    "x_shape": list(x.shape),
    "sliced_shape": list(sliced.shape),
    "loss": loss.tolist(),
    "x.grad_shape": list(x.grad.shape) if x.grad is not None else None,
    "x.grad_nonzero": bool((x.grad != 0).any()) if x.grad is not None else False,
    "status": "OK"
}
print(json.dumps(out, indent=2))
