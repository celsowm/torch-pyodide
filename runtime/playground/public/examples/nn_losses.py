import json
import torch.nn.functional as F

logits = torch.randn((2, 3))
target = torch.tensor([0, 2])
loss = F.cross_entropy(logits, target)
mse = F.mse_loss(logits, torch.zeros((2, 3)))
out = {
    "cross_entropy": loss.tolist(),
    "mse_loss": mse.tolist(),
}
print(json.dumps(out, indent=2))
