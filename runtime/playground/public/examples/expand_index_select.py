import json
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
e = x.expand(3, 2, 2)
indices = torch.tensor([0, 1], dtype=torch.int64)
s = torch.index_select(x, 0, indices)
out = {
  "input": x.tolist(),
  "expand_shape": list(e.shape),
  "expand": e.tolist(),
  "index_select": s.tolist(),
}
print(json.dumps(out, indent=2))
