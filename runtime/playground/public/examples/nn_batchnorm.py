import json
import torch.nn as nn

bn = nn.BatchNorm1d(4)
x = torch.randn((3, 4))
y = bn(x)
out = {"shape": list(y.shape), "values": y.tolist()}
print(json.dumps(out, indent=2))
