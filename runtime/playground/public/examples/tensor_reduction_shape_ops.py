import json
import torch

torch.manual_seed(0)

x = torch.tensor([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]])

# ---------- dim-based max/min (values + indices) ----------
vmax, imax = torch.max(x, dim=1)
max_match = vmax[0].item() == 5.0 and vmax[1].item() == 6.0 and imax[0].item() == 1 and imax[1].item() == 2
vmin, imin = x.min(dim=0)
min_match = vmin[0].item() == 1.0 and vmin[2].item() == 2.0

# ---------- amax / amin / aminmax ----------
amax_all = torch.amax(x).item() == 6.0
amin_row = torch.amin(x, dim=1)
amin_match = amin_row[0].item() == 1.0 and amin_row[1].item() == 3.0
lo, hi = torch.aminmax(x)
aminmax_match = lo.item() == 1.0 and hi.item() == 6.0

# ---------- logsumexp ----------
lse = torch.logsumexp(x, dim=1)
import math
ref0 = math.log(math.exp(1.0) + math.exp(5.0) + math.exp(2.0))
logsumexp_match = abs(lse[0].item() - ref0) < 1e-3

# ---------- var / std (correction) ----------
v = torch.tensor([1.0, 2.0, 3.0, 4.0])
var_unb = torch.var(v).item()
var_biased = torch.var(v, correction=0).item()
var_match = abs(var_unb - 1.6666666) < 1e-3 and abs(var_biased - 1.25) < 1e-3
std_match = abs(torch.std(v).item() - var_unb ** 0.5) < 1e-4

# ---------- count_nonzero ----------
cnz = torch.count_nonzero(torch.tensor([0.0, 1.0, 0.0, 2.0, 3.0])).item()
cnz_match = cnz == 3

# ---------- nan_to_num ----------
nt = torch.nan_to_num(torch.tensor([1.0, float("inf"), float("-inf")]), posinf=100.0, neginf=-100.0)
nan_to_num_match = nt[0].item() == 1.0 and nt[1].item() == 100.0 and nt[2].item() == -100.0

# ---------- shape ops ----------
unb = torch.unbind(x, dim=0)
unbind_match = len(unb) == 2 and list(unb[0].shape) == [3]
mv = torch.movedim(torch.randn(2, 3, 4), 0, 2)
movedim_match = list(mv.shape) == [3, 4, 2]
sw = torch.swapaxes(torch.randn(2, 3), 0, 1)
swap_match = list(sw.shape) == [3, 2]
rv = torch.ravel(x)
ravel_match = list(rv.shape) == [6]
bt = torch.broadcast_to(torch.tensor([1.0, 2.0, 3.0]), [2, 3])
broadcast_match = list(bt.shape) == [2, 3] and bt[1, 2].item() == 3.0

# ---------- stacking ----------
a1 = torch.tensor([1.0, 2.0])
a2 = torch.tensor([3.0, 4.0])
hs = torch.hstack([a1, a2])
hstack_match = list(hs.shape) == [4] and hs[3].item() == 4.0
vs = torch.vstack([a1, a2])
vstack_match = list(vs.shape) == [2, 2]
cs = torch.column_stack([a1, a2])
colstack_match = list(cs.shape) == [2, 2] and cs[0, 1].item() == 3.0

# ---------- flip helpers ----------
fu = torch.flipud(x)
flipud_match = fu[0, 0].item() == 4.0
fl = torch.fliplr(x)
fliplr_match = fl[0, 0].item() == 2.0

# ---------- diff / trace / diagflat ----------
df = torch.diff(torch.tensor([1.0, 3.0, 6.0, 10.0]))
diff_match = df[0].item() == 2.0 and df[2].item() == 4.0
tr = torch.trace(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
trace_match = tr.item() == 5.0
dfl = torch.diagflat(torch.tensor([1.0, 2.0, 3.0]))
diagflat_match = list(dfl.shape) == [3, 3] and dfl[1, 1].item() == 2.0

# ---------- inner / vdot / kron / tensordot / dist ----------
inr = torch.inner(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]))
inner_match = abs(inr.item() - 32.0) < 1e-4
vd = torch.vdot(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
vdot_match = abs(vd.item() - 11.0) < 1e-4
kr = torch.kron(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]))
kron_match = list(kr.shape) == [4] and kr[2].item() == 2.0
td = torch.tensordot(torch.randn(2, 3, 4), torch.randn(4, 5), dims=([2], [0]))
tensordot_match = list(td.shape) == [2, 3, 5]
dst = torch.dist(torch.tensor([0.0, 0.0]), torch.tensor([3.0, 4.0]))
dist_match = abs(dst.item() - 5.0) < 1e-4

# ---------- take_along_dim / clip ----------
tad = torch.take_along_dim(x, torch.argmax(x, dim=1, keepdim=True), dim=1)
take_match = tad[0, 0].item() == 5.0 and tad[1, 0].item() == 6.0
clp = torch.clip(torch.tensor([-2.0, 0.5, 3.0]), min=0.0, max=1.0)
clip_match = clp[0].item() == 0.0 and clp[2].item() == 1.0

out = {
    "max_match": max_match,
    "min_match": min_match,
    "amax_all": amax_all,
    "amin_match": amin_match,
    "aminmax_match": aminmax_match,
    "logsumexp_match": logsumexp_match,
    "var_match": var_match,
    "std_match": std_match,
    "count_nonzero_match": cnz_match,
    "nan_to_num_match": nan_to_num_match,
    "unbind_match": unbind_match,
    "movedim_match": movedim_match,
    "swap_match": swap_match,
    "ravel_match": ravel_match,
    "broadcast_match": broadcast_match,
    "hstack_match": hstack_match,
    "vstack_match": vstack_match,
    "colstack_match": colstack_match,
    "flipud_match": flipud_match,
    "fliplr_match": fliplr_match,
    "diff_match": diff_match,
    "trace_match": trace_match,
    "diagflat_match": diagflat_match,
    "inner_match": inner_match,
    "vdot_match": vdot_match,
    "kron_match": kron_match,
    "tensordot_match": tensordot_match,
    "dist_match": dist_match,
    "take_match": take_match,
    "clip_match": clip_match,
    "status": "OK",
}
failed = [k for k, v in out.items() if k != "status" and not v]
assert not failed, f"mismatches: {failed}"
print(json.dumps(out, indent=2))
