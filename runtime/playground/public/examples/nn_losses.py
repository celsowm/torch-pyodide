import json
import torch
import torch.nn.functional as F

logits = torch.tensor([[0.5, -0.3, 1.2], [-0.8, 0.1, 0.9]])
target = torch.tensor([0, 2])
loss_ce = F.cross_entropy(logits, target)
loss_mse = F.mse_loss(logits, torch.zeros((2, 3)))

probs = torch.tensor([0.5, 0.3])
binary_target = torch.tensor([1.0, 0.0])
loss_bce = F.binary_cross_entropy(probs, binary_target)

logits_bce = torch.tensor([0.5, -0.3])
loss_bce_logits = F.binary_cross_entropy_with_logits(logits_bce, binary_target)

a = torch.tensor([0.5, 0.3])
z = torch.tensor([0.0, 0.0])
loss_l1 = F.l1_loss(a, z)
loss_smooth_l1 = F.smooth_l1_loss(torch.tensor([0.5, -0.2]), z)

out = {
"cross_entropy": loss_ce.tolist(),
"mse_loss": loss_mse.tolist(),
"binary_cross_entropy": loss_bce.tolist(),
"binary_cross_entropy_with_logits": loss_bce_logits.tolist(),
"l1_loss": loss_l1.tolist(),
"smooth_l1_loss": loss_smooth_l1.tolist(),
}
print(json.dumps(out, indent=2))
