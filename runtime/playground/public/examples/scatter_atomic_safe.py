import json

import torch

torch.manual_seed(7)

V, D = 5, 4
weight = torch.randn(V, D, requires_grad=True)
# duplicate token ids (id 2 appears three times, id 0 twice) to expose the
# non-atomic scatter_add race. padding_idx=1 must be skipped.
indices = torch.tensor([2, 0, 2, 1, 2, 0], dtype=torch.long)
grad_output = torch.randn(6, D)

out = torch.embedding(indices, weight, padding_idx=1)
out.backward(grad_output)
gpu_grad = weight.grad.detach().clone()

# Reference computed on CPU by hand (the correct, race-free result).
ref = [[0.0] * D for _ in range(V)]
for i in range(6):
    tok = int(indices[i])
    if tok == 1:
        continue
    for d in range(D):
        ref[tok][d] += float(grad_output[i, d])

max_diff = 0.0
for v in range(V):
    for d in range(D):
        max_diff = max(max_diff, abs(gpu_grad[v, d].item() - ref[v][d]))

# Also exercise index_select backward with duplicate indices.
x = torch.randn(4, 3, requires_grad=True)
idx = torch.tensor([3, 0, 3, 1, 3], dtype=torch.long)  # index 3 duplicated
y = torch.index_select(x, 0, idx)
g = torch.randn(5, 3)
y.backward(g)
gpu_x = x.grad.detach().clone()
ref_x = [[0.0] * 3 for _ in range(4)]
for j in range(5):
    src = int(idx[j])
    for d in range(3):
        ref_x[src][d] += float(g[j, d])
max_diff = max(max_diff, max(abs(gpu_x[a, b].item() - ref_x[a][b]) for a in range(4) for b in range(3)))

ok = max_diff < 1e-4
print(json.dumps({"ok": ok, "max_diff": max_diff}, sort_keys=True))
