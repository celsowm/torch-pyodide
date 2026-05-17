import json
import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
t = a.T
r = t.reshape((4,))
out = {
  "transpose": t.tolist(),
  "reshape": r.tolist(),
}
print(json.dumps(out, indent=2))
