import json
import torch

# masked_select with autograd
torch.manual_seed(42)

x = torch.randn((3, 4), requires_grad=True)
mask = torch.tensor([[True, False, True, False],
                     [False, True, False, True],
                     [True, True, False, False]])
selected = torch.masked_select(x, mask)

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
