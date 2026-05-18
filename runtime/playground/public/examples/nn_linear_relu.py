import json
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
x = torch.randn((3, 4))
y = model(x)
out = {"shape": list(y.shape), "values": y.tolist()}
print(json.dumps(out, indent=2))
