import json
import torch
import torch.nn as nn

torch.manual_seed(0)

# --- torch.pdist (GPU pdist.wgsl) ---
x = torch.randn(5, 3)
dist = torch.pdist(x, p=2)
# Reference: pairwise Euclidean distances over the upper triangle.
ref = []
for i in range(5):
    for j in range(i + 1, 5):
        ref.append((x[i] - x[j]).pow(2).sum().sqrt().item())
pdist_match = all(abs(dist[k].item() - ref[k]) < 1e-4 for k in range(len(ref)))

# --- nn.init.orthogonal_ (GPU gram_schmidt.wgsl) ---
w = torch.empty(4, 4)
nn.init.orthogonal_(w)
gram = w.matmul(w.t())  # should be close to identity (orthonormal rows)
ortho_err = (gram - torch.eye(4)).abs().max().item()

# --- torch.linalg.matrix_rank (GPU where+sum) ---
m = torch.tensor([[3.0, 0.0], [0.0, 0.0]])
rank = torch.linalg.matrix_rank(m)

# --- einsum trace ii-> (GPU diag) ---
a = torch.randn(3, 3)
trace = torch.einsum("ii->", a)
trace_match = abs(trace.item() - a.diag().sum().item()) < 1e-5

# --- Categorical.sample (GPU argmax) ---
cat = torch.distributions.Categorical(torch.tensor([0.1, 0.2, 0.7]))
cat_samples = torch.stack([cat.sample() for _ in range(20)])
cat_in_range = bool(((cat_samples >= 0) & (cat_samples < 3)).all().item())

out = {
    "pdist_len": len(dist),
    "pdist_match": pdist_match,
    "ortho_max_err": ortho_err,
    "matrix_rank": int(rank),
    "einsum_trace_match": trace_match,
    "categorical_in_range": cat_in_range,
    "status": "OK",
}
print(json.dumps(out, indent=2))
