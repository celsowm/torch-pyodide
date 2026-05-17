import json
import torch

x = torch.tensor([[1.0, 3.0, 2.0], [9.0, 0.5, 7.0]])
max_idx = torch.argmax(x)
min_idx = torch.argmin(x)
out = {
  "input": x.tolist(),
  "argmax": max_idx.tolist(),
  "argmin": min_idx.tolist(),
}
print(json.dumps(out, indent=2))
