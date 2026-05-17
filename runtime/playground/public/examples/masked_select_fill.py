import json
import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
filled = torch.masked_fill(x, mask, 99.0)
selected = torch.masked_select(x, mask)
out = {
  "input": x.tolist(),
  "mask": mask.tolist(),
  "masked_fill": filled.tolist(),
  "masked_select": selected.tolist(),
}
print(json.dumps(out, indent=2))
