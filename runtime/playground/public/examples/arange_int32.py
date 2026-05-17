import json
import torch

x = torch.arange(1, 10, 2, dtype="int32")
out = {
  "dtype": x.dtype,
  "values": x.tolist(),
}
print(json.dumps(out, indent=2))
