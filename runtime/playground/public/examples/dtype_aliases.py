import json
import torch

idx = torch.tensor([10, 11, 12], dtype=torch.long)
roundtrip = idx.float().long()
mask = torch.tensor([True, False, True], dtype=torch.bool)
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
selected = torch.masked_select(x, mask)

out = {
  "long_alias": str(torch.long).replace("torch.", ""),
  "bool_alias": str(torch.bool).replace("torch.", ""),
  "double_alias": str(torch.double).replace("torch.", ""),
  "half_alias": str(torch.half).replace("torch.", ""),
  "int_alias": str(torch.int).replace("torch.", ""),
  "float_alias": str(torch.float).replace("torch.", ""),
  "short_alias": str(torch.short).replace("torch.", ""),
  "int8_alias": str(torch.int8).replace("torch.", ""),
  "uint8_alias": str(torch.uint8).replace("torch.", ""),
  "idx_dtype": str(idx.dtype).replace("torch.", ""),
  "roundtrip_dtype": str(roundtrip.dtype).replace("torch.", ""),
  "mask_dtype": str(mask.dtype).replace("torch.", ""),
  "selected": selected.tolist(),
}

print(json.dumps(out, indent=2))
