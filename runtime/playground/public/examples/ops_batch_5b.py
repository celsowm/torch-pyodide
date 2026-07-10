import json

import torch

torch.manual_seed(3)

max_diff = 0.0


def check(name, got, ref):
    global max_diff
    d = max(abs(a - b) for a, b in zip(got, ref))
    max_diff = max(max_diff, d)


# --- index_copy / index_add / index_fill ---
base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
idx = torch.tensor([1, 0], dtype=torch.long)
src = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

ic = torch.index_copy(base, 0, idx, src)
ic_ref = base.tolist()
for r, i in enumerate(idx.tolist()):
    ic_ref[i] = src.tolist()[r]
check("index_copy", [v for row in ic.tolist() for v in row],
      [v for row in ic_ref for v in row])

ia = torch.index_add(base, 0, idx, src)
ia_ref = [row[:] for row in base.tolist()]
for r, i in enumerate(idx.tolist()):
    for c in range(3):
        ia_ref[i][c] += src.tolist()[r][c]
check("index_add", [v for row in ia.tolist() for v in row],
      [v for row in ia_ref for v in row])

ifl = torch.index_fill(base, 1, torch.tensor([0, 2], dtype=torch.long), 9.0)
ifl_ref = [row[:] for row in base.tolist()]
for r in range(2):
    for c, ci in enumerate([0, 2]):
        ifl_ref[r][ci] = 9.0
check("index_fill", [v for row in ifl.tolist() for v in row],
      [v for row in ifl_ref for v in row])

# --- take ---
t = torch.arange(0, 12, 1.0).reshape(3, 4)
tk = torch.tensor([0, 5, 11], dtype=torch.long)
tv = torch.take(t, tk)
tv_ref = [t.tolist()[0][0], t.tolist()[1][1], t.tolist()[2][3]]
check("take", tv.tolist(), tv_ref)

# --- unfold ---
u = torch.arange(0, 8, 1.0)
uf = u.unfold(0, 3, 2)
# windows starting at 0 and 2 -> [0,1,2],[2,3,4]; plus start 4 -> [4,5,6]
uf_ref = [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
check("unfold", [v for w in uf.tolist() for v in w],
      [v for w in uf_ref for v in w])

# --- cdist ---
a = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
b = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
cd = torch.cdist(a, b)
cd_ref = []
for ra in a.tolist():
    for rb in b.tolist():
        cd_ref.append((sum((x - y) ** 2 for x, y in zip(ra, rb))) ** 0.5)
check("cdist", [v for row in cd.tolist() for v in row], cd_ref)

# --- pdist ---
pd = torch.pdist(a)
ap = a.tolist()
pd_ref = []
for i in range(len(ap)):
    for j in range(i + 1, len(ap)):
        pd_ref.append((sum((x - y) ** 2 for x, y in zip(ap[i], ap[j]))) ** 0.5)
check("pdist", pd.tolist(), pd_ref)

# --- qr (self-consistency) ---
x = torch.randn(4, 3)
Q, R = torch.linalg.qr(x)
recon = (Q.matmul(R) - x).abs().max().item()
ortho = (Q.transpose(0, 1).matmul(Q) - torch.eye(3)).abs().max().item()
qr_max = max(recon, ortho)
max_diff = max(max_diff, qr_max)

ok = max_diff < 1e-3
print(json.dumps({"ok": ok, "max_diff": max_diff}, sort_keys=True))
