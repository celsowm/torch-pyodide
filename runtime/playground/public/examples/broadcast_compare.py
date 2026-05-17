import json
import torch

a = torch.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
threshold = torch.tensor([3.0, 3.0, 3.0])
out = {
  "a": a.tolist(),
  "threshold": threshold.tolist(),
  "a > threshold": torch.gt(a, threshold).tolist(),
  "a <= threshold": torch.le(a, threshold).tolist(),
  "where(a > 3, a, 0)": torch.where(torch.gt(a, torch.tensor(3.0)), a, torch.zeros_like(a)).tolist(),
}
print(json.dumps(out, indent=2))
