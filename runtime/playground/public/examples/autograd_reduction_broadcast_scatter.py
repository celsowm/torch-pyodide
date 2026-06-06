import json

import torch

# Reduction backward by dimension
m = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
s = m.sum(dim=1)
mean0 = m.mean(dim=0)

# Broadcast backward
row = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
(col := None)
(col := torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True))
scalar = torch.tensor(2.0, requires_grad=True)

# Boolean condition where backward
cond = torch.tensor([True, False, True], dtype=torch.bool)
xb = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
yb = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Scatter backward
base = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
index = torch.tensor([0, 2], dtype=torch.int64)
src = torch.tensor([10.0, 20.0], requires_grad=True)

# Now run all backward passes
s.sum().backward()
grad_sum_dim = m.grad.tolist()
m.grad = None

mean0.sum().backward()
grad_mean_dim = m.grad.tolist()
m.grad = None

(row + torch.ones([2, 3])).sum().backward()
grad_broadcast_row = row.grad.tolist()

(col * torch.ones([3, 4])).sum().backward()
grad_broadcast_col = col.grad.tolist()

(scalar + torch.ones([2, 3])).sum().backward()
grad_broadcast_scalar = scalar.grad.tolist()

torch.where(cond, xb, yb).sum().backward()
grad_where_bool_x = xb.grad.tolist()
grad_where_bool_y = yb.grad.tolist()

torch.scatter(base, 0, index, src).sum().backward()
grad_scatter_base = base.grad.tolist()
grad_scatter_src = src.grad.tolist()

print(json.dumps({
    "grad_sum_dim": [[round(v, 4) for v in row] for row in grad_sum_dim],
    "grad_mean_dim": [[round(v, 4) for v in row] for row in grad_mean_dim],
    "grad_broadcast_row": [[round(v, 4) for v in row] for row in grad_broadcast_row],
    "grad_broadcast_col": [[round(v, 4) for v in row] for row in grad_broadcast_col],
    "grad_broadcast_scalar": round(grad_broadcast_scalar, 4),
    "grad_where_bool_x": [round(v, 4) for v in grad_where_bool_x],
    "grad_where_bool_y": [round(v, 4) for v in grad_where_bool_y],
    "grad_scatter_base": [round(v, 4) for v in grad_scatter_base],
    "grad_scatter_src": [round(v, 4) for v in grad_scatter_src],
}, sort_keys=True))
