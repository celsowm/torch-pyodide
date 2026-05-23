import json
import torch

idx = torch.tensor([10, 11, 12], dtype=torch.long)
roundtrip = idx.float().long()
mask = torch.tensor([True, False, True], dtype=torch.bool)
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
selected = torch.masked_select(x, mask)

out = {
  "long_alias": torch.long,
  "bool_alias": torch.bool,
  "double_alias": torch.double,
  "half_alias": torch.half,
  "int_alias": torch.int,
  "float_alias": torch.float,
  "short_alias": torch.short,
  "char_alias": torch.char,
  "byte_alias": torch.byte,
  "idx_dtype": idx.dtype,
  "roundtrip_dtype": roundtrip.dtype,
  "mask_dtype": mask.dtype,
  "selected": selected.tolist(),
}

print(json.dumps(out, indent=2))
