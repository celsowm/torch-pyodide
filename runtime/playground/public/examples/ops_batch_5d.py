import json

import torch

torch.manual_seed(7)

max_diff = 0.0
worst = ""


def check(name, d):
    global max_diff, worst
    if d > max_diff:
        max_diff = d
        worst = name


n = 5
A = torch.randn(n, n)
A = (A + A.T) * 0.5

# --- eigh (symmetric) ---
ev, V = torch.linalg.eigh(A)
rev, _ = torch.linalg.eigh(A)
check("eigh_eig", max(abs(a - b) for a, b in zip(sorted(ev.tolist()), sorted(rev.tolist()))))
recon = V.matmul(torch.diag(ev)).matmul(V.permute([1, 0]))
check("eigh_recon", max(abs(a - b) for a, b in zip(A.reshape(-1).tolist(), recon.reshape(-1).tolist())))
I = V.matmul(V.permute([1, 0]))
check("eigh_ortho", max(abs(a - b) for a, b in zip(torch.eye(n).reshape(-1).tolist(), I.reshape(-1).tolist())))

# --- svd (general m x n) ---
m, k = 4, 3
B = torch.randn(m, k)
U, S, Vh = torch.linalg.svd(B)
_, reS, _ = torch.linalg.svd(B)
check("svd_S", max(abs(a - b) for a, b in zip(sorted(S.tolist(), reverse=True), sorted(reS.tolist(), reverse=True))))
recon_svd = U.matmul(torch.diag(S)).matmul(Vh)
check("svd_recon", max(abs(a - b) for a, b in zip(B.reshape(-1).tolist(), recon_svd.reshape(-1).tolist())))
check("svd_Uortho", max(abs(a - b) for a, b in zip(torch.eye(k).reshape(-1).tolist(), (U.permute([1, 0]).matmul(U)).reshape(-1).tolist())))
check("svd_Vortho", max(abs(a - b) for a, b in zip(torch.eye(k).reshape(-1).tolist(), (Vh.matmul(Vh.permute([1, 0]))).reshape(-1).tolist())))

# --- eig (symmetric -> real eigenvalues) ---
w, _ = torch.linalg.eig(A)
rw, _ = torch.linalg.eig(A)
w_real = [v.real if isinstance(v, complex) else float(v) for v in w.tolist()]
rw_real = [v.real if isinstance(v, complex) else float(v) for v in rw.tolist()]
check("eig_eig", max(abs(a - b) for a, b in zip(sorted(w_real), sorted(rw_real))))

ok = max_diff < 1e-2
print(json.dumps({"ok": ok, "max_diff": max_diff, "worst": worst}, sort_keys=True))
