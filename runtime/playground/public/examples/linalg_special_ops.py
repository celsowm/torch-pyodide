import json
import torch

torch.manual_seed(0)

# ---------- torch.linalg extensions (GPU-composed) ----------
a = torch.randn(4, 3)
b = torch.randn(4, 3)
cr = torch.linalg.cross(a, b)
# reference cross product via components
cx = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
cross_match = (cr[:, 0] - cx).abs().max().item() < 1e-4

m = torch.randn(3, 3)
sign, logabs = torch.linalg.slogdet(m)
det = torch.linalg.det(m)
slogdet_match = abs((sign * torch.exp(logabs)).item() - det.item()) < 1e-3

sv = torch.linalg.svdvals(m)
_, s_ref, _ = torch.linalg.svd(m)
svdvals_match = (sv - s_ref).abs().max().item() < 1e-4

x = torch.randn(4, 5)
d0 = torch.linalg.diagonal(x)
diag_ref = torch.stack([x[i, i] for i in range(4)])
diag_match = (d0 - diag_ref).abs().max().item() < 1e-5

v = torch.randn(6)
vn2 = torch.linalg.vector_norm(v, 2)
vn2_match = abs(vn2.item() - v.pow(2).sum().sqrt().item()) < 1e-4
vn1 = torch.linalg.vector_norm(v, 1)
vn1_match = abs(vn1.item() - v.abs().sum().item()) < 1e-4

mn_fro = torch.linalg.matrix_norm(m, "fro")
mn_fro_match = abs(mn_fro.item() - m.pow(2).sum().sqrt().item()) < 1e-3

vv = torch.tensor([1.0, 2.0, 3.0])
van = torch.linalg.vander(vv, 3)
vander_match = bool(
    (van[1, 0].item() == 4.0) and (van[1, 1].item() == 2.0) and (van[1, 2].item() == 1.0)
)

md = torch.linalg.multi_dot([torch.randn(2, 3), torch.randn(3, 4), torch.randn(4, 2)])
multidot_ok = list(md.shape) == [2, 2]

# lu_factor / lu_solve round-trip
A = torch.randn(3, 3)
B = torch.randn(3, 2)
LU, piv = torch.linalg.lu_factor(A)
X = torch.linalg.lu_solve(LU, piv, B)
lu_solve_resid = (A.matmul(X) - B).abs().max().item()

# matrix_exp of a small matrix (scaling & squaring)
me = torch.linalg.matrix_exp(m * 0.05)
matexp_finite = bool(torch.isfinite(me).all().item())

# ---------- torch.special (GPU elementwise) ----------
sx = torch.tensor([0.5, 1.0, 2.0])
entr = torch.special.entr(torch.tensor([0.0, 0.5, 1.0]))
entr_match = abs(entr[1].item() - (-0.5 * torch.log(torch.tensor(0.5)).item())) < 1e-4
gl = torch.special.gammaln(torch.tensor([1.0, 2.0, 3.0, 4.0]))
gammaln_match = abs(gl[3].item() - 1.7917594) < 1e-3
ndtr = torch.special.ndtr(torch.tensor([0.0]))
ndtr_match = abs(ndtr[0].item() - 0.5) < 1e-5
sinc = torch.special.sinc(torch.tensor([0.0, 0.5]))
sinc_match = abs(sinc[0].item() - 1.0) < 1e-5 and abs(sinc[1].item() - 0.6366198) < 1e-3
logit = torch.special.logit(torch.tensor([0.5]))
logit_match = abs(logit[0].item()) < 1e-5

out = {
    "cross_match": cross_match,
    "slogdet_match": slogdet_match,
    "svdvals_match": svdvals_match,
    "diagonal_match": diag_match,
    "vector_norm_match": bool(vn2_match and vn1_match),
    "matrix_norm_match": mn_fro_match,
    "vander_match": vander_match,
    "multidot_ok": multidot_ok,
    "lu_solve_resid": lu_solve_resid,
    "matrix_exp_finite": matexp_finite,
    "entr_match": entr_match,
    "gammaln_match": gammaln_match,
    "ndtr_match": ndtr_match,
    "sinc_match": sinc_match,
    "logit_match": logit_match,
    "lu_solve_ok": bool(lu_solve_resid < 1e-3),
    "status": "OK",
}
print(json.dumps(out, indent=2))
assert lu_solve_resid < 1e-3, f"lu_solve residual too large: {lu_solve_resid}"
