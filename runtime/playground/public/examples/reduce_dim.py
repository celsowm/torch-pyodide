import json
import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
out = {
  "input": x.tolist(),
  "sum(0)": torch.sum(x, 0).tolist(),
  "sum(1)": torch.sum(x, 1).tolist(),
  "mean(0)": torch.mean(x, 0).tolist(),
  "mean(1)": torch.mean(x, 1).tolist(),
  "prod": torch.prod(x).tolist(),
  "min": torch.min(x).tolist(),
  "max": torch.max(x).tolist(),
}
print(json.dumps(out, indent=2))
