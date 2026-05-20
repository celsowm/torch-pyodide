import torch

# Cat + Expand + Where + Scatter backward
a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
b = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

c = torch.cat([a, b])
e = torch.expand(a, [3, 3])
print("cat:", c.tolist())
print("expand:", e.tolist())

loss = c.sum() + e.sum()
loss.backward()
print("grad a:", a.grad.tolist())
print("grad b:", b.grad.tolist())

# Where
cond = torch.tensor([1.0, 0.0, 1.0])
x = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
w = torch.where(cond, x, y)
print("where:", w.tolist())

x.grad = None; y.grad = None
w.sum().backward()
print("grad x:", x.grad.tolist())
print("grad y:", y.grad.tolist())
