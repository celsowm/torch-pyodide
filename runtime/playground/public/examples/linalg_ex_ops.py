import json

import torch

torch.manual_seed(3)

max_diff = 0.0
worst = ""


def check(name, d):
    global max_diff, worst
    if d > max_diff:
        max_diff = d
        worst = name


# ---------- cholesky_ex ----------
A = torch.randn(4, 4)
A = A.matmul(A.T) + torch.eye(4)  # SPD
L, info = torch.linalg.cholesky_ex(A)
check("cholesky_ex_info", abs(int(info.item())))
check("cholesky_ex_lower", (L - torch.tril(L)).abs().max().item())
check("cholesky_ex_recon", (L.matmul(L.T) - A).abs().max().item())

# ---------- inv_ex ----------
M = torch.randn(4, 4)
Minv, info = torch.linalg.inv_ex(M)
check("inv_ex_info", abs(int(info.item())))
check("inv_ex_roundtrip", (M.matmul(Minv) - torch.eye(4)).abs().max().item())

# ---------- lu_factor_ex ----------
LU, piv, info = torch.linalg.lu_factor_ex(M)
check("lu_factor_ex_info", abs(int(info.item())))
LU0, piv0 = torch.linalg.lu_factor(M)
check("lu_factor_ex_matches", (LU - LU0).abs().max().item())
check("lu_factor_ex_pivots", (piv - piv0).abs().max().item())
X = torch.linalg.lu_solve(LU, piv, torch.eye(4))
check("lu_factor_ex_solve", (M.matmul(X) - torch.eye(4)).abs().max().item())

# ---------- solve_ex ----------
B = torch.randn(4, 3)
Xs, info = torch.linalg.solve_ex(M, B)
check("solve_ex_info", abs(int(info.item())))
check("solve_ex_resid", (M.matmul(Xs) - B).abs().max().item())

# ---------- ldl_factor / ldl_factor_ex ----------
A2 = torch.randn(4, 4)
A2 = A2.matmul(A2.T) + torch.eye(4)  # SPD -> valid LDL^T
LD, piv = torch.linalg.ldl_factor(A2)
# Reconstruct A = L @ diag(D) @ L^T from the packed LD factor using plain
# Python so the check exercises ldl_factor's output directly.
nn = 4
ld = LD.tolist()
Am = A2.tolist()
Lm = [[ld[i][j] if i > j else (1.0 if i == j else 0.0) for j in range(nn)] for i in range(nn)]
Dm = [ld[i][i] for i in range(nn)]
ldl_err = 0.0
for i in range(nn):
    for j in range(nn):
        acc = 0.0
        for p in range(nn):
            acc += Lm[i][p] * Dm[p] * Lm[j][p]
        ldl_err = max(ldl_err, abs(acc - Am[i][j]))
check("ldl_factor_recon", ldl_err)
LDx, pivx, infox = torch.linalg.ldl_factor_ex(A2)
check("ldl_factor_ex_info", abs(int(infox.item())))
check("ldl_factor_ex_matches", (LDx - LD).abs().max().item())

# ---------- ldl_solve ----------
Y = torch.randn(4, 2)
Xl = torch.linalg.ldl_solve(LD, piv, Y)
check("ldl_solve_resid", (A2.matmul(Xl) - Y).abs().max().item())

# ---------- householder_product (orthonormality of columns) ----------
m, n = 5, 4
H = torch.randn(m, n)
k = min(m, n)
# Build proper Householder coefficients tau_i = 2 / (v_i^T v_i) from each column
# (v_i has an implicit leading 1, matching the convention the op expects).
taus = []
for i in range(k):
    col = H[i:, i]
    v = torch.cat([torch.ones(1), col[1:]])
    taus.append(2.0 / (v * v).sum().item())
tau = torch.tensor(taus)
Q = torch.linalg.householder_product(H, tau)
check("householder_shape", max(abs(v - w) for v, w in zip(Q.shape, [m, n])))
QtQ = Q.T.matmul(Q)
check("householder_ortho", (QtQ - torch.eye(n)).abs().max().item())

# ---------- vecdot ----------
x = torch.randn(5)
y = torch.randn(5)
check("vecdot_real", abs(torch.linalg.vecdot(x, y).item() - (x * y).sum().item()))

# ---------- tensorinv ----------
T = torch.randn(2, 2, 2, 2)
Tinv = torch.linalg.tensorinv(T, 2)
Tmat = T.reshape(4, 4)
I4 = Tmat.matmul(Tinv.reshape(4, 4))
check("tensorinv", (I4 - torch.eye(4)).abs().max().item())

# ---------- tensorsolve ----------
S = torch.randn(3, 3, 3, 3)
R = torch.randn(3, 3)
Xr = torch.linalg.tensorsolve(S, R, 2)
Smat = S.reshape(9, 9)
check("tensorsolve", (Smat.matmul(Xr.reshape(9, 1)) - R.reshape(9, 1)).abs().max().item())

ok = max_diff < 1e-2
assert ok, f"max_diff {max_diff} worst {worst}"
print(json.dumps({"ok": ok, "max_diff": max_diff, "worst": worst}, sort_keys=True))
