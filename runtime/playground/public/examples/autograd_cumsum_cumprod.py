import json

import torch

torch.manual_seed(42)
y = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
tr = y.tril()
tu = y.triu()
loss2 = tr.sum() + tu.sum()
y.grad = None
loss2.backward()
grad_2d = [[round(v, 4) for v in row] for row in y.grad.tolist()]

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
cs = x.cumsum(0).tolist()
cp = x.cumprod(0).tolist()

print(json.dumps({
"tril": tr.tolist(),
"triu": tu.tolist(),
"grad_2d": grad_2d,
"cumsum": cs,
"cumprod": cp,
}, sort_keys=True))
