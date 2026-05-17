import json
import torch

a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = torch.tensor([10.0, 20.0, 30.0])
c = torch.tensor([[1.0], [2.0]])
out = {
  "a + b (broadcast)": a.add(b).tolist(),
  "a * c (broadcast)": a.mul(c).tolist(),
  "b - a (broadcast)": b.sub(a).tolist(),
  "shape_a": list(a.shape),
  "shape_b": list(b.shape),
}
print(json.dumps(out, indent=2))
