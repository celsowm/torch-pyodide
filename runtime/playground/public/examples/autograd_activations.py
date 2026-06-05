import torch
import torch.nn.functional as F

# Activation backward: sigmoid, tanh, gelu, silu, leaky_relu, softmax
x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], requires_grad=True)

for name, fn in [
    ("sigmoid", lambda: F.sigmoid(x)),
    ("tanh", lambda: F.tanh(x)),
    ("gelu", lambda: F.gelu(x)),
    ("silu", lambda: F.silu(x)),
    ("leaky_relu", lambda: F.leaky_relu(x, 0.1)),
    ("softmax", lambda: F.softmax(x, dim=-1)),
]:
    y = fn()
    loss = y.sum()
    x.grad = None
    loss.backward()
    print(f"{name:12s} grad: {x.grad.tolist()}")
