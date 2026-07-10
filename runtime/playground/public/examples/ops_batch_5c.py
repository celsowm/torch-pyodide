import json

import torch

torch.manual_seed(5)

max_diff = 0.0


def check(name, got, ref, tol=1e-3):
    global max_diff
    d = max(abs(a - b) for a, b in zip(got, ref))
    max_diff = max(max_diff, d)


def check_eq(name, got, ref):
    check(name, got, ref, tol=0.0)


# --- searchsorted ---
sorted_seq = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
values = torch.tensor([0.0, 2.0, 3.0, 6.0, 10.0])
ss = torch.searchsorted(sorted_seq, values)
check_eq("searchsorted", ss.tolist(), [0, 1, 1, 3, 5])
ssr = torch.searchsorted(sorted_seq, values, right=True)
check_eq("searchsorted_right", ssr.tolist(), [0, 1, 2, 3, 5])

# --- kthvalue (1-D and along dim) ---
kv = torch.tensor([7.0, 2.0, 5.0, 1.0, 9.0])
val, idx = torch.kthvalue(kv, 3)
check_eq("kthvalue_val", [val.item()], [sorted(kv.tolist())[2]])
check_eq("kthvalue_idx", [idx.item()], [sorted(range(5), key=lambda i: kv.tolist()[i])[2]])

mat = torch.tensor([[3.0, 1.0, 4.0], [2.0, 8.0, 5.0]])
v2, i2 = torch.kthvalue(mat, 2, dim=1)
check_eq("kthvalue_dim_val", v2.tolist(), [sorted(r)[1] for r in mat.tolist()])
check_eq("kthvalue_dim_idx", i2.tolist(), [sorted(range(3), key=lambda c: mat.tolist()[r][c])[1] for r in range(2)])

# --- median (odd length to match definition) ---
md = torch.tensor([8.0, 3.0, 1.0, 5.0, 2.0])
m = torch.median(md)
check_eq("median", [m.item()], [sorted(md.tolist())[2]])

# --- quantile ---
q = torch.tensor([0.25, 0.5, 0.75])
qv = torch.quantile(md, q)
s = sorted(md.tolist())
n = len(s)
qs = [0.25, 0.5, 0.75]
qref = []
for qq in qs:
    pos = (n - 1) * qq
    k0 = int(pos)
    k1 = min(n - 1, k0 + 1)
    qref.append(s[k0] + (s[k1] - s[k0]) * (pos - k0))
check("quantile", qv.tolist(), qref)

# --- mode (dim=None and along dim) ---
mode_in = torch.tensor([4.0, 4.0, 4.0, 1.0, 2.0, 1.0, 3.0])
mv, mc = torch.mode(mode_in)
counts = {}
for v in mode_in.tolist():
    counts[v] = counts.get(v, 0) + 1
mref = max(counts.items(), key=lambda kv: kv[1])
check_eq("mode_val", [mv.item()], [mref[0]])
check_eq("mode_count", [mc.item()], [mref[1]])

mm = torch.tensor([[1.0, 1.0, 2.0], [3.0, 3.0, 3.0]])
vm, cm = torch.mode(mm, dim=1)
vm_ref = [max({v: r.count(v) for v in set(r)}.items(), key=lambda kv: kv[1])[0] for r in mm.tolist()]
cm_ref = [max({v: r.count(v) for v in set(r)}.values()) for r in mm.tolist()]
check_eq("mode_dim_val", vm.tolist(), vm_ref)
check_eq("mode_dim_count", cm.tolist(), cm_ref)

# --- unique ---
u = torch.tensor([3.0, 1.0, 3.0, 2.0, 1.0, 3.0])
uu = torch.unique(u)
check_eq("unique", uu.tolist(), sorted(set(u.tolist())))
uu2, uc2 = torch.unique(u, return_counts=True)
check_eq("unique_vals", uu2.tolist(), sorted(set(u.tolist())))
cref = sorted((v, u.tolist().count(v)) for v in set(u.tolist()))
check_eq("unique_counts", uc2.tolist(), [c for _, c in cref])

# --- histogram ---
h, edges = torch.histogram(torch.tensor([0.5, 1.5, 1.5, 3.0, 4.5]), bins=4, range=(0.0, 5.0))
check_eq("histogram", h.tolist(), [1, 2, 1, 1])
edges_ref = [0.0 + i * 1.25 for i in range(5)]
check("histogram_edges", edges.tolist(), edges_ref)

ok = max_diff < 1e-3
print(json.dumps({"ok": ok, "max_diff": max_diff}, sort_keys=True))
