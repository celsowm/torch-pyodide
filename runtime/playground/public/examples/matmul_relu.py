import json
import torch

x = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
w = torch.tensor([[0.5, 1.0], [1.5, -1.0]])
y = x.matmul(w).relu()
out = {
  "shape": list(y.shape),
  "values": y.tolist(),
  "sum": y.sum().tolist(),
}
print(json.dumps(out, indent=2))
