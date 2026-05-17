import json
import torch

x = torch.tensor([[1.0, 4.0], [9.0, 16.0]])
y = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
z = torch.tensor([[1.0, 2.718281828], [7.389056099, 1.0]])
out = {
  "sqrt": torch.sqrt(x).tolist(),
  "exp": torch.exp(y).tolist(),
  "log": torch.log(z).tolist(),
}
print(json.dumps(out, indent=2))
