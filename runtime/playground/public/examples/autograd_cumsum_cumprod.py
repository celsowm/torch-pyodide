import torch

# Cumsum + Cumprod + Tril + Triu + Flip backward
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

cs = x.cumsum(0)
cp = x.cumprod(0)
print("cumsum:", cs.tolist())
print("cumprod:", cp.tolist())

loss = cs.sum() + cp.sum()
loss.backward()
print("grad:", x.grad.tolist())

# 2d ops
y = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
tr = y.tril()
tu = y.triu()
print("tril:", tr.tolist())
print("triu:", tu.tolist())

loss2 = tr.sum() + tu.sum()
y.grad = None
loss2.backward()
print("grad 2d:", y.grad.tolist())
