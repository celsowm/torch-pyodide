import json
import torch

# Conv2d with autograd (runtime-backed backprop)
torch.manual_seed(42)

# Input: batch=1, channels=1, height=5, width=5
x = torch.randn((1, 1, 5, 5), requires_grad=True)

# Kernel: out_channels=1, in_channels=1, kh=3, kw=3
weight = torch.ones((1, 1, 3, 3), requires_grad=True)
bias = torch.zeros((1,), requires_grad=True)

# Forward: conv2d
out_conv = torch.nn.functional.conv2d(x, weight, bias, padding=1)

# Loss: sum of output
loss = out_conv.sum()

# Backward
loss.backward()

out = {
    "output_shape": list(out_conv.shape),
    "loss": loss.tolist(),
    "weight_grad_shape": list(weight.grad.shape) if weight.grad is not None else None,
    "weight_grad_sum": weight.grad.sum().tolist() if weight.grad is not None else None,
    "bias_grad": bias.grad.tolist() if bias.grad is not None else None,
    "input_grad_shape": list(x.grad.shape) if x.grad is not None else None
}
print(json.dumps(out, indent=2))
