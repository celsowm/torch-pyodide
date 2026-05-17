import json
import torch

cond = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
x = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
out_tensor = torch.where(cond, x, y)
out = {
  "condition": cond.tolist(),
  "x": x.tolist(),
  "y": y.tolist(),
  "where": out_tensor.tolist(),
}
print(json.dumps(out, indent=2))
