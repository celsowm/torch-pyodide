import json
import torch

x = torch.randn((2, 4))
out = {
  "shape": list(x.shape),
  "values": x.tolist(),
}
print(json.dumps(out, indent=2))
