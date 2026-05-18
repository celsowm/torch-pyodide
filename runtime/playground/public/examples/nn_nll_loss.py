import json
import torch
import torch.nn.functional as F

# Example: NLL Loss with LogSoftmax
logits = torch.randn((2, 4))
target = torch.tensor([1, 3])

# Manual nll_loss on log_softmax input
log_probs = F.log_softmax(logits, dim=-1)
nll = F.nll_loss(log_probs, target, reduction="mean")

# Cross entropy = log_softmax + nll_loss
ce = F.cross_entropy(logits, target, reduction="mean")

# Compare batch elements
nll_none = F.nll_loss(log_probs, target, reduction="none")

out = {
    "log_probs": [[round(v, 4) for v in row] for row in log_probs.tolist()],
    "nll_loss_mean": round(nll.tolist()[0], 4),
    "cross_entropy_mean": round(ce.tolist()[0], 4),
    "nll_loss_none": [round(v, 4) for v in nll_none.tolist()],
}
print(json.dumps(out, indent=2))
