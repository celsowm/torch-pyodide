import json
import torch.nn as nn

conv = nn.Conv2d(1, 4, 3)
x = torch.randn((2, 1, 6, 6))
y = conv(x)
out = {"shape": list(y.shape), "sum": y.sum().tolist()}
print(json.dumps(out, indent=2))
