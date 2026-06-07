import json
import torch
import torch.nn as nn

torch.manual_seed(0)

# 1. GroupNorm forward + per-(N, G) stats
gn = nn.GroupNorm(num_groups=2, num_channels=6)
x = torch.randn(2, 6, 4, 4)
y = gn(x)
forward_shape = list(y.shape)

import torch.nn.functional as F
y_manual = F.group_norm(x, 2, gn.weight, gn.bias, gn.eps)
y_diff = (y - y_manual).abs().max().item()

y_list = y.tolist()
per_group_means = []
per_group_vars = []
for n in range(2):
    for g in range(2):
        sub = [y_list[n][g * 3 + i][r][c] for i in range(3) for r in range(4) for c in range(4)]
        m = sum(sub) / len(sub)
        v = sum((vi - m) ** 2 for vi in sub) / len(sub)
        per_group_means.append(m)
        per_group_vars.append(v)

max_per_group_mean = max(abs(m) for m in per_group_means)

# 2. Autograd flow
x2 = torch.randn(2, 6, 4, 4, requires_grad=True)
y2 = gn(x2)
loss = (y2 * y2).sum()
try:
    loss.backward()
    grad_shape = list(x2.grad.shape)
    backward_ok = True
except Exception:
    import traceback
    backward_traceback = traceback.format_exc()
    grad_shape = []
    backward_ok = False

if backward_ok:
    flat_grad = []
    def _flatten(t):
        if isinstance(t, list):
            for v in t:
                _flatten(v)
        else:
            flat_grad.append(t)
    _flatten(x2.grad.tolist())
    grad_nonzero = any(abs(v) > 1e-6 for v in flat_grad)
else:
    grad_nonzero = False

# 3. InstanceNorm1d / 2d
in1 = nn.InstanceNorm1d(6, affine=True)
in2 = nn.InstanceNorm2d(8, affine=True)
in1_shape = list(in1(torch.randn(2, 6, 10)).shape)
in2_shape = list(in2(torch.randn(2, 8, 5, 5)).shape)

# 4. state_dict
gn_keys = sorted(gn.state_dict().keys())
in1_keys = sorted(in1.state_dict().keys())
in2_keys = sorted(in2.state_dict().keys())

result = {
    "forward_shape": forward_shape,
    "grad_shape": grad_shape,
    "grad_nonzero": bool(grad_nonzero),
    "in1_shape": in1_shape,
    "in2_shape": in2_shape,
    "gn_state_dict_keys": gn_keys,
    "in1_state_dict_keys": in1_keys,
    "in2_state_dict_keys": in2_keys,
    "max_per_group_mean": max_per_group_mean,
    "y_diff_vs_manual": y_diff,
    "per_group_means": per_group_means,
    "per_group_vars": per_group_vars,
}
if not backward_ok:
    result["backward_traceback"] = backward_traceback
print(json.dumps(result))
