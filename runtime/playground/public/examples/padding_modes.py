import json
import torch

x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
x2 = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])

c = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="constant", value=0.0)
p = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate")
r = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect")
circ = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="circular")

c2 = torch.nn.functional.pad(x2, (1, 1, 1, 1), mode="constant", value=0.0)
p2 = torch.nn.functional.pad(x2, (1, 1, 1, 1), mode="replicate")
r2 = torch.nn.functional.pad(x2, (1, 1, 1, 1), mode="reflect")
circ2 = torch.nn.functional.pad(x2, (1, 1, 1, 1), mode="circular")

out = {
"original_shape": list(x.shape),
"original2_shape": list(x2.shape),
"constant_values": c.tolist(),
"replicate_values": p.tolist(),
"reflect_values": r.tolist(),
"circular_values": circ.tolist(),
"constant2_values": c2.tolist(),
"replicate2_values": p2.tolist(),
"reflect2_values": r2.tolist(),
"circular2_values": circ2.tolist(),
}
print(json.dumps(out, indent=2))
