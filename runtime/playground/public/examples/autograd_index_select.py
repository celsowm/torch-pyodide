import json
import torch

# index_select with autograd
torch.manual_seed(42)

x = torch.randn((5, 3), requires_grad=True)
idx = torch.tensor([0, 2, 4], dtype=torch.int32)
selected = torch.index_select(x, 0, idx)

loss = selected.sum()
loss.backward()

out = {
    "x_shape": list(x.shape),
    "selected_shape": list(selected.shape),
    "loss": loss.tolist(),
    "x.grad_nonzero": bool((x.grad != 0).any()) if x.grad is not None else False,
    "x.grad_shape": list(x.grad.shape) if x.grad is not None else None,
}
print(json.dumps(out, indent=2))
