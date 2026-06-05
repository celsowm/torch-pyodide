import json
import torch
import torch.nn as nn

# BatchNorm in training vs eval mode
bn = nn.BatchNorm1d(4)
bn.weight = nn.Parameter(torch.ones((4,)))
bn.bias = nn.Parameter(torch.zeros((4,)))

x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0],
                  [9.0, 10.0, 11.0, 12.0]])

# Training mode: normalizes using batch statistics
bn.train()
y_train = bn(x)

# Eval mode: normalizes using running_mean/running_var
bn.eval()
y_eval = bn(x)

# After training pass, running_mean/running_var are updated
running_mean = bn.running_mean.tolist()
running_var = bn.running_var.tolist()

out = {
    "x_shape": list(x.shape),
    "output_train": [[round(v, 4) for v in row] for row in y_train.tolist()],
    "output_eval": [[round(v, 4) for v in row] for row in y_eval.tolist()],
    "running_mean": [round(v, 4) for v in running_mean],
    "running_var": [round(v, 4) for v in running_var],
}
print(json.dumps(out, indent=2))
