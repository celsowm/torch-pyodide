import torch

# Activation backward: sigmoid, tanh, gelu, silu, leaky_relu, softmax
x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], requires_grad=True)

for name, fn in [
    ("sigmoid", x.sigmoid),
    ("tanh", x.tanh),
    ("gelu", x.gelu),
    ("silu", x.silu),
    ("leaky_relu", lambda: x.leaky_relu(0.1)),
    ("softmax", lambda: x.softmax(dim=-1)),
]:
    y = fn()
    loss = y.sum()
    x.grad = None
    loss.backward()
    print(f"{name:12s} grad: {x.grad.tolist()}")
