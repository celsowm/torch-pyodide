import json
import torch

x = torch.rand((2, 3))
out = {
  "shape": list(x.shape),
  "values": x.tolist(),
}
print(json.dumps(out, indent=2))
