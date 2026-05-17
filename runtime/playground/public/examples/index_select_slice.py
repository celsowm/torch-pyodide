import json
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
out = {
  "select_row_1": torch.select(x, 0, 1).tolist(),
  "slice_rows_0_3_step2": torch.slice(x, 0, 0, 3, 2).tolist(),
  "getitem_row_1": x[1].tolist(),
}
print(json.dumps(out, indent=2))
