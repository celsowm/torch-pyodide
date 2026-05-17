import json
import torch

x = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]])
out = {
  "input": x.tolist(),
  "sigmoid": torch.sigmoid(x).tolist(),
  "tanh": torch.tanh(x).tolist(),
  "sin": torch.sin(x).tolist(),
  "cos": torch.cos(x).tolist(),
  "gelu": torch.gelu(x).tolist(),
  "silu": torch.silu(x).tolist(),
  "leaky_relu": torch.leaky_relu(x, 0.01).tolist(),
  "floor": torch.floor(x).tolist(),
  "ceil": torch.ceil(x).tolist(),
  "round": torch.round(x).tolist(),
  "reciprocal": torch.reciprocal(x).tolist(),
  "square": torch.square(x).tolist(),
}
print(json.dumps(out, indent=2))
