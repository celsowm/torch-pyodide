import json
import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
out = {
  "input_shape": list(x.shape),
  "sum(0, keepdim)": torch.sum(x, 0, True).tolist(),
  "sum(0, keepdim)_shape": list(torch.sum(x, 0, True).shape),
  "mean(1, keepdim)": torch.mean(x, 1, True).tolist(),
  "mean(1, keepdim)_shape": list(torch.mean(x, 1, True).shape),
}
print(json.dumps(out, indent=2))
