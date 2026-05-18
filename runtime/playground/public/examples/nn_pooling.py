import json
import torch
import torch.nn as nn

x = torch.randn((1, 2, 8, 8))
max_pool = nn.MaxPool2d(2)
avg_pool = nn.AvgPool2d(2)
out = {
    "max_pool_shape": list(max_pool(x).shape),
    "avg_pool_shape": list(avg_pool(x).shape),
}
print(json.dumps(out, indent=2))
