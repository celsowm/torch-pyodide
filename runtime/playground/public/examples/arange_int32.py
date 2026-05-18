import json
import torch

x = torch.arange(1, 10, 2, dtype=torch.int32)
out = {
  "dtype": str(x.dtype),
  "values": x.tolist(),
}
print(json.dumps(out, indent=2))
