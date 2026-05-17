import json
import torch

x = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])
out = {
  "input": x.tolist(),
  "abs": torch.abs(x).tolist(),
  "neg": torch.neg(x).tolist(),
}
print(json.dumps(out, indent=2))
