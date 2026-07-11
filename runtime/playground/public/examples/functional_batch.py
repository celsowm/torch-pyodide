import json
import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)

# ---------- activations ----------
ls = F.logsigmoid(torch.tensor([0.0]))
logsigmoid_match = abs(ls[0].item() - math.log(0.5)) < 1e-4

sm = F.softmin(torch.tensor([1.0, 2.0, 3.0]))
softmin_match = sm[0].item() > sm[2].item() and abs(sm.sum().item() - 1.0) < 1e-4

ht = F.hardtanh(torch.tensor([-2.0, 0.5, 3.0]))
hardtanh_match = ht[0].item() == -1.0 and ht[1].item() == 0.5 and ht[2].item() == 1.0

r6 = F.relu6(torch.tensor([-1.0, 3.0, 8.0]))
relu6_match = r6[0].item() == 0.0 and r6[1].item() == 3.0 and r6[2].item() == 6.0

hs = F.hardshrink(torch.tensor([0.3, 0.7, -0.8]))
hardshrink_match = hs[0].item() == 0.0 and abs(hs[1].item() - 0.7) < 1e-5 and abs(hs[2].item() + 0.8) < 1e-5

ss = F.softshrink(torch.tensor([0.7, -0.7, 0.3]))
softshrink_match = abs(ss[0].item() - 0.2) < 1e-5 and abs(ss[1].item() + 0.2) < 1e-5 and ss[2].item() == 0.0

th = F.threshold(torch.tensor([-1.0, 2.0]), 0.0, -5.0)
threshold_match = th[0].item() == -5.0 and th[1].item() == 2.0

se = F.selu(torch.tensor([0.0, 1.0]))
selu_match = abs(se[0].item()) < 1e-6 and abs(se[1].item() - 1.05070098) < 1e-4

gs = F.gumbel_softmax(torch.tensor([[1.0, 2.0, 3.0]]), tau=1.0, hard=True)
gumbel_match = abs(gs.sum().item() - 1.0) < 1e-4

# ---------- distances ----------
cs = F.cosine_similarity(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), dim=1)
cosine_match = abs(cs[0].item() - 1.0) < 1e-4
pw = F.pairwise_distance(torch.tensor([[0.0, 0.0]]), torch.tensor([[3.0, 4.0]]))
pairwise_match = abs(pw[0].item() - 5.0) < 1e-2
pd = F.pdist(torch.tensor([[0.0, 0.0], [3.0, 4.0]]))
pdist_match = abs(pd[0].item() - 5.0) < 1e-3

# ---------- rms_norm ----------
rn = F.rms_norm(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), (4,))
ms = (1.0 + 4.0 + 9.0 + 16.0) / 4.0
rms_match = abs(rn[0, 0].item() - 1.0 / math.sqrt(ms + 1e-6)) < 1e-3

# ---------- scaled_dot_product_attention ----------
q = torch.randn(2, 4, 8)
k = torch.randn(2, 4, 8)
v = torch.randn(2, 4, 8)
attn = F.scaled_dot_product_attention(q, k, v)
sdpa_match = list(attn.shape) == [2, 4, 8]
attn_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
sdpa_causal_match = list(attn_causal.shape) == [2, 4, 8]

# ---------- pixel shuffle round-trip ----------
x = torch.randn(1, 8, 4, 4)
ps = F.pixel_shuffle(x, 2)
pixel_shuffle_match = list(ps.shape) == [1, 2, 8, 8]
pu = F.pixel_unshuffle(ps, 2)
pixel_roundtrip_match = list(pu.shape) == [1, 8, 4, 4] and (pu - x).abs().max().item() < 1e-4

# ---------- losses ----------
kl = F.kl_div(torch.tensor([[-1.0, -1.0]]), torch.tensor([[0.5, 0.5]]), reduction="batchmean")
kl_match = bool(torch.isfinite(kl).item())
sml = F.soft_margin_loss(torch.tensor([1.0, -1.0]), torch.tensor([1.0, 1.0]))
soft_margin_match = bool(torch.isfinite(sml).item())
hel = F.hinge_embedding_loss(torch.tensor([0.5, 2.0]), torch.tensor([1.0, -1.0]))
hinge_match = bool(torch.isfinite(hel).item())
mrl = F.margin_ranking_loss(torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([1.0]), margin=0.5)
mr_match = abs(mrl.item()) < 1e-6
cel = F.cosine_embedding_loss(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), torch.tensor([1.0]))
cosine_emb_match = abs(cel.item()) < 1e-4
pnl = F.poisson_nll_loss(torch.tensor([0.0]), torch.tensor([1.0]))
poisson_match = bool(torch.isfinite(pnl).item())
tml = F.triplet_margin_loss(torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4))
triplet_match = bool(torch.isfinite(tml).item())

# ---------- pooling / upsample aliases ----------
aap = F.adaptive_avg_pool1d(torch.randn(1, 3, 8), 1)
adaptive_match = list(aap.shape) == [1, 3, 1]
up = F.upsample_nearest(torch.randn(1, 1, 2, 2), scale_factor=2)
upsample_match = list(up.shape) == [1, 1, 4, 4]

out = {
    "logsigmoid_match": logsigmoid_match,
    "softmin_match": softmin_match,
    "hardtanh_match": hardtanh_match,
    "relu6_match": relu6_match,
    "hardshrink_match": hardshrink_match,
    "softshrink_match": softshrink_match,
    "threshold_match": threshold_match,
    "selu_match": selu_match,
    "gumbel_match": gumbel_match,
    "cosine_match": cosine_match,
    "pairwise_match": pairwise_match,
    "pdist_match": pdist_match,
    "rms_match": rms_match,
    "sdpa_match": sdpa_match,
    "sdpa_causal_match": sdpa_causal_match,
    "pixel_shuffle_match": pixel_shuffle_match,
    "pixel_roundtrip_match": pixel_roundtrip_match,
    "kl_match": kl_match,
    "soft_margin_match": soft_margin_match,
    "hinge_match": hinge_match,
    "mr_match": mr_match,
    "cosine_emb_match": cosine_emb_match,
    "poisson_match": poisson_match,
    "triplet_match": triplet_match,
    "adaptive_match": adaptive_match,
    "upsample_match": upsample_match,
    "status": "OK",
}
failed = [key for key, val in out.items() if key != "status" and not val]
assert not failed, f"mismatches: {failed}"
print(json.dumps(out, indent=2))
