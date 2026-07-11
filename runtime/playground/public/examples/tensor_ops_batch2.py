import json
import torch

torch.manual_seed(0)

# ---------- sign / positive / negative ----------
sb = torch.signbit(torch.tensor([-1.0, 0.5, -3.0]))
signbit_match = bool(sb[0].item()) and (not bool(sb[1].item())) and bool(sb[2].item())
neg_match = torch.negative(torch.tensor([1.0, -2.0]))[0].item() == -1.0
pos_match = torch.positive(torch.tensor([1.0, -2.0]))[1].item() == -2.0

# ---------- nansum / nanmean ----------
nan = float("nan")
ns = torch.nansum(torch.tensor([1.0, nan, 3.0]))
nansum_match = abs(ns.item() - 4.0) < 1e-5
nm = torch.nanmean(torch.tensor([2.0, nan, 4.0]))
nanmean_match = abs(nm.item() - 3.0) < 1e-5

# ---------- cross ----------
a = torch.tensor([1.0, 0.0, 0.0])
b = torch.tensor([0.0, 1.0, 0.0])
cr = torch.cross(a, b)
cross_match = cr[0].item() == 0.0 and cr[1].item() == 0.0 and cr[2].item() == 1.0

# ---------- rot90 ----------
m = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
r90 = torch.rot90(m, 1, (0, 1))
rot90_match = r90[0, 0].item() == 2.0 and r90[0, 1].item() == 4.0 and r90[1, 0].item() == 1.0

# ---------- tensor_split / hsplit / vsplit ----------
ts = torch.tensor_split(torch.arange(10, dtype=torch.float32), 3)
tensor_split_match = list(ts[0].shape) == [4] and list(ts[1].shape) == [3] and list(ts[2].shape) == [3]
hs = torch.hsplit(torch.arange(6, dtype=torch.float32), 2)
hsplit_match = list(hs[0].shape) == [3] and hs[1][0].item() == 3.0

# ---------- addmm / addmv / baddbmm / addbmm ----------
inp = torch.zeros(2, 2)
m1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
m2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
am = torch.addmm(inp, m1, m2)
addmm_match = am[0, 0].item() == 1.0 and am[1, 1].item() == 4.0
bmat1 = torch.randn(3, 2, 4)
bmat2 = torch.randn(3, 4, 5)
bad = torch.baddbmm(torch.zeros(3, 2, 5), bmat1, bmat2)
baddbmm_match = list(bad.shape) == [3, 2, 5]
adb = torch.addbmm(torch.zeros(2, 5), bmat1, bmat2)
addbmm_match = list(adb.shape) == [2, 5]

# ---------- chain_matmul ----------
cm = torch.chain_matmul(torch.randn(2, 3), torch.randn(3, 4), torch.randn(4, 2))
chain_match = list(cm.shape) == [2, 2]

# ---------- meshgrid / cartesian_prod ----------
gx, gy = torch.meshgrid(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0]), indexing="ij")
meshgrid_match = list(gx.shape) == [3, 2] and gx[2, 0].item() == 3.0 and gy[0, 1].item() == 5.0
cp = torch.cartesian_prod(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
cartesian_match = list(cp.shape) == [4, 2] and cp[3, 0].item() == 2.0 and cp[3, 1].item() == 4.0

# ---------- diag_embed / block_diag ----------
de = torch.diag_embed(torch.tensor([1.0, 2.0, 3.0]))
diag_embed_match = list(de.shape) == [3, 3] and de[1, 1].item() == 2.0 and de[0, 1].item() == 0.0
bd = torch.block_diag(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0]]))
block_diag_match = list(bd.shape) == [3, 3] and bd[2, 2].item() == 5.0 and bd[0, 2].item() == 0.0

# ---------- tril_indices / triu_indices ----------
tri = torch.tril_indices(3, 3)
tril_idx_match = list(tri.shape) == [2, 6]
triu = torch.triu_indices(3, 3)
triu_idx_match = list(triu.shape) == [2, 6]

# ---------- trapezoid ----------
tz = torch.trapezoid(torch.tensor([1.0, 2.0, 3.0]))
trapezoid_match = abs(tz.item() - 4.0) < 1e-5
tzx = torch.trapezoid(torch.tensor([1.0, 2.0, 3.0]), x=torch.tensor([0.0, 1.0, 2.0]))
trapezoid_x_match = abs(tzx.item() - 4.0) < 1e-5

# ---------- renorm ----------
rn = torch.renorm(torch.tensor([[3.0, 4.0], [6.0, 8.0]]), 2, 0, 5.0)
renorm_match = abs((rn[0, 0].item() ** 2 + rn[0, 1].item() ** 2) ** 0.5 - 5.0) < 1e-3

# ---------- sinc / isreal ----------
sc = torch.sinc(torch.tensor([0.0, 0.5]))
sinc_match = abs(sc[0].item() - 1.0) < 1e-5
ir = torch.isreal(torch.tensor([1.0, 2.0]))
isreal_match = bool(ir[0].item()) and bool(ir[1].item())

# ---------- Tensor-method parity ----------
method_match = (
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]).trace().item() == 5.0
    and torch.tensor([1.0, 2.0, 3.0]).inner(torch.tensor([1.0, 1.0, 1.0])).item() == 6.0
    and torch.tensor([-1.0, 2.0]).clamp_min(0.0)[0].item() == 0.0
)

out = {
    "signbit_match": signbit_match,
    "neg_match": bool(neg_match),
    "pos_match": bool(pos_match),
    "nansum_match": nansum_match,
    "nanmean_match": nanmean_match,
    "cross_match": cross_match,
    "rot90_match": rot90_match,
    "tensor_split_match": tensor_split_match,
    "hsplit_match": hsplit_match,
    "addmm_match": addmm_match,
    "baddbmm_match": baddbmm_match,
    "addbmm_match": addbmm_match,
    "chain_match": chain_match,
    "meshgrid_match": meshgrid_match,
    "cartesian_match": cartesian_match,
    "diag_embed_match": diag_embed_match,
    "block_diag_match": block_diag_match,
    "tril_idx_match": tril_idx_match,
    "triu_idx_match": triu_idx_match,
    "trapezoid_match": trapezoid_match,
    "trapezoid_x_match": trapezoid_x_match,
    "renorm_match": renorm_match,
    "sinc_match": sinc_match,
    "isreal_match": isreal_match,
    "method_match": method_match,
    "status": "OK",
}
failed = [k for k, v in out.items() if k != "status" and not v]
assert not failed, f"mismatches: {failed}"
print(json.dumps(out, indent=2))
