import json
import torch

x = torch.tensor([[-1.0, 0.2, 0.8], [1.5, 2.0, -0.3]])
y = torch.clamp(x, 0.0, 1.0)
out = {
  "input": x.tolist(),
  "clamped": y.tolist(),
}
print(json.dumps(out, indent=2))
