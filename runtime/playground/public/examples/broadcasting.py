import json
import torch

a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = torch.tensor([10.0, 20.0, 30.0])
out = {
  "a + b (broadcast)": a.add(b).tolist(),
  "a * 2.0 (scalar broadcast)": a.mul(2.0).tolist(),
  "shape_a": list(a.shape),
  "shape_b": list(b.shape),
}
print(json.dumps(out, indent=2))
