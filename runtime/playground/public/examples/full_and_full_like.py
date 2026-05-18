import json
import torch

base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
a = torch.full((2, 2), 7.0, dtype=torch.int32)
b = torch.full_like(base, 9.0)
out = {
  "full": a.tolist(),
  "full_dtype": str(a.dtype),
  "full_like": b.tolist(),
  "full_like_dtype": str(b.dtype),
}
print(json.dumps(out, indent=2))
