import json
import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
cat_rows = torch.cat([a, b], dim=0)
cat_cols = torch.cat([a, b], dim=1)
stacked = torch.stack([a, b], dim=0)
out = {
  "cat_dim0": cat_rows.tolist(),
  "cat_dim0_shape": list(cat_rows.shape),
  "cat_dim1": cat_cols.tolist(),
  "cat_dim1_shape": list(cat_cols.shape),
  "stack_dim0": stacked.tolist(),
  "stack_dim0_shape": list(stacked.shape),
}
print(json.dumps(out, indent=2))
