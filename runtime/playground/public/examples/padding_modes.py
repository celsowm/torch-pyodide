import json
import torch

# Padding modes: constant, reflect, replicate, circular
# Real PyTorch requires 3D or 4D input for reflect/replicate/circular with 4-tuple padding.
x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

# Constant padding
c = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="constant", value=0.0)

# Reflect padding
r = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect")

# Replicate padding
p = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate")

# Circular padding
circ = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="circular")

out = {
    "original_shape": list(x.shape),
    "constant_shape": list(c.shape),
    "constant_values": c.tolist(),
    "reflect_shape": list(r.shape),
    "reflect_values": r.tolist(),
    "replicate_shape": list(p.shape),
    "replicate_values": p.tolist(),
    "circular_shape": list(circ.shape),
    "circular_values": circ.tolist()
}
print(json.dumps(out, indent=2))
