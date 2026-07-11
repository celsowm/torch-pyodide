import torch
import torch.nn as nn

torch.manual_seed(0)

checks = {}

x = torch.tensor([[-2.0, 0.5, 3.0]])

# activation modules mirror their functional counterparts
checks["ReLU6"] = nn.ReLU6()(x)[0, 2].item() == 3.0
checks["Hardtanh"] = nn.Hardtanh()(x)[0, 0].item() == -1.0
checks["Hardshrink"] = nn.Hardshrink()(torch.tensor([0.3, 0.7]))[0].item() == 0.0
checks["Softshrink"] = abs(nn.Softshrink()(torch.tensor([0.7]))[0].item() - 0.2) < 1e-5
checks["Threshold"] = nn.Threshold(0.0, -5.0)(x)[0, 0].item() == -5.0
checks["Softmin"] = abs(nn.Softmin()(torch.tensor([1.0, 2.0, 3.0])).sum().item() - 1.0) < 1e-4
checks["LogSigmoid"] = nn.LogSigmoid()(torch.tensor([0.0]))[0].item() < 0.0
checks["SELU"] = nn.SELU()(torch.tensor([1.0]))[0].item() > 1.0
checks["Softplus"] = nn.Softplus()(torch.tensor([0.0]))[0].item() > 0.0
checks["Softsign"] = abs(nn.Softsign()(torch.tensor([1.0]))[0].item() - 0.5) < 1e-5
checks["Tanhshrink"] = nn.Tanhshrink()(torch.tensor([0.0]))[0].item() == 0.0
checks["Mish"] = nn.Mish()(torch.tensor([0.0]))[0].item() == 0.0
checks["Hardswish"] = nn.Hardswish()(torch.tensor([0.0]))[0].item() == 0.0
checks["Hardsigmoid"] = abs(nn.Hardsigmoid()(torch.tensor([0.0]))[0].item() - 0.5) < 1e-5

# Softmax2d over channel dim
y = torch.randn(2, 3, 4, 4)
sm2 = nn.Softmax2d()(y)
checks["Softmax2d"] = abs(sm2[0, :, 0, 0].sum().item() - 1.0) < 1e-4

# distances
a = torch.randn(4, 8)
b = torch.randn(4, 8)
cs = nn.CosineSimilarity(dim=1)(a, b)
checks["CosineSimilarity"] = cs.shape[0] == 4 and cs.max().item() <= 1.0001
pd = nn.PairwiseDistance()(a, b)
checks["PairwiseDistance"] = pd.shape[0] == 4 and pd.min().item() >= 0.0

# RMSNorm
rn = nn.RMSNorm(8)
out = rn(a)
checks["RMSNorm"] = list(out.shape) == [4, 8]

# pixel shuffle round-trip
z = torch.randn(1, 8, 4, 4)
ps = nn.PixelShuffle(2)(z)
pu = nn.PixelUnshuffle(2)(ps)
checks["PixelShuffle_shape"] = list(ps.shape) == [1, 2, 8, 8]
checks["PixelShuffle_roundtrip"] = (pu - z).abs().max().item() < 1e-4

# upsampling aliases
img = torch.randn(1, 1, 2, 2)
checks["UpsamplingNearest2d"] = list(nn.UpsamplingNearest2d(scale_factor=2)(img).shape) == [1, 1, 4, 4]
checks["UpsamplingBilinear2d"] = list(nn.UpsamplingBilinear2d(scale_factor=2)(img).shape) == [1, 1, 4, 4]

# losses
logits = torch.randn(3, 5)
target = torch.tensor([0, 2, 4])
log_probs = torch.log_softmax(logits, dim=1)
checks["NLLLoss"] = nn.NLLLoss()(log_probs, target).item() > 0.0

probs = torch.sigmoid(torch.randn(6))
tgt = (torch.randn(6) > 0).float()
checks["BCELoss"] = nn.BCELoss()(probs, tgt).item() > 0.0

p = torch.log_softmax(torch.randn(3, 4), dim=1)
q = torch.softmax(torch.randn(3, 4), dim=1)
checks["KLDivLoss"] = nn.KLDivLoss(reduction="batchmean")(p, q).item() is not None

inp = torch.randn(5)
smt = (torch.randn(5) > 0).float() * 2 - 1
checks["SoftMarginLoss"] = nn.SoftMarginLoss()(inp, smt).item() > 0.0

checks["HingeEmbeddingLoss"] = nn.HingeEmbeddingLoss()(torch.randn(5), smt).item() is not None

i1 = torch.randn(4)
i2 = torch.randn(4)
mt = (torch.randn(4) > 0).float() * 2 - 1
checks["MarginRankingLoss"] = nn.MarginRankingLoss(margin=0.5)(i1, i2, mt).item() >= 0.0

e1 = torch.randn(3, 8)
e2 = torch.randn(3, 8)
cet = torch.tensor([1.0, -1.0, 1.0])
checks["CosineEmbeddingLoss"] = nn.CosineEmbeddingLoss()(e1, e2, cet).item() is not None

pred = torch.abs(torch.randn(5)) + 0.1
pt = torch.abs(torch.randn(5))
checks["PoissonNLLLoss"] = nn.PoissonNLLLoss(log_input=False)(pred, pt).item() is not None

anc = torch.randn(3, 8)
pos = torch.randn(3, 8)
neg = torch.randn(3, 8)
checks["TripletMarginLoss"] = nn.TripletMarginLoss()(anc, pos, neg).item() >= 0.0

failed = [k for k, v in checks.items() if not v]
print("passed:", len(checks) - len(failed), "/", len(checks))
if failed:
    print("FAILED:", failed)
assert not failed
print("nn_modules_batch OK")
