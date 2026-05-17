import json
import torch

a = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
b = torch.tensor([[1.0, 3.0], [3.0, 4.0]])
out = {
  "a": a.tolist(),
  "b": b.tolist(),
  "eq": torch.eq(a, b).tolist(),
  "ne": torch.ne(a, b).tolist(),
  "lt": torch.lt(a, b).tolist(),
  "le": torch.le(a, b).tolist(),
  "gt": torch.gt(a, b).tolist(),
  "ge": torch.ge(a, b).tolist(),
}
print(json.dumps(out, indent=2))
