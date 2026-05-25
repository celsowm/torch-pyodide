import torch

# Reduction backward by dimension
m = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
s = m.sum(dim=1)
print("sum dim:", s.tolist())
s.sum().backward()
print("grad sum dim:", m.grad.tolist())

m.grad = None
mean0 = m.mean(dim=0)
print("mean dim:", mean0.tolist())
mean0.sum().backward()
print("grad mean dim:", m.grad.tolist())

# Broadcast backward
row = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
(row + torch.ones([2, 3])).sum().backward()
print("grad broadcast row:", row.grad.tolist())

col = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
(col * torch.ones([3, 4])).sum().backward()
print("grad broadcast col:", col.grad.tolist())

scalar = torch.tensor(2.0, requires_grad=True)
(scalar + torch.ones([2, 3])).sum().backward()
print("grad broadcast scalar:", scalar.grad.tolist())

# Boolean condition where backward
cond = torch.tensor([True, False, True], dtype=torch.bool)
x = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
torch.where(cond, x, y).sum().backward()
print("grad where bool x:", x.grad.tolist())
print("grad where bool y:", y.grad.tolist())

# Scatter backward
base = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
index = torch.tensor([0, 2], dtype=torch.int64)
src = torch.tensor([10.0, 20.0], requires_grad=True)
torch.scatter(base, 0, index, src).sum().backward()
print("grad scatter base:", base.grad.tolist())
print("grad scatter src:", src.grad.tolist())
