import json
import torch

x = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
out = {
  "input_shape": list(x.shape),
  "flatten_1_2": torch.flatten(x, 1, 2).tolist(),
  "squeeze_all_shape": list(torch.squeeze(x).shape),
  "unsqueeze_dim0_shape": list(torch.unsqueeze(torch.tensor([1.0, 2.0]), 0).shape),
}
print(json.dumps(out, indent=2))
