import json
import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.ones((2, 2))
c = a.add(b).mul(torch.tensor([[2.0, 2.0], [2.0, 2.0]]))
out = {
  "shape": list(c.shape),
  "values": c.tolist(),
  "sum": c.sum().tolist(),
  "mean": c.mean().tolist(),
  "cuda_available": torch.cuda.is_available(),
  "cuda_device_count": torch.cuda.device_count(),
}
print(json.dumps(out, indent=2))
