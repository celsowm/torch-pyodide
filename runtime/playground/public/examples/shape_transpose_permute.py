import json
import torch

x = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
t = torch.transpose(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 0, 1)
p = torch.permute(x, (2, 0, 1))
out = {
  "transpose": t.tolist(),
  "permute_shape": list(p.shape),
}
print(json.dumps(out, indent=2))
