import json
import torch

# NLL Loss + Cross Entropy with autograd (runtime-backed backprop)
torch.manual_seed(42)

# NLL Loss
logits = torch.tensor([
    [-2.0, -1.0, 0.0],  # class 2 is correct
    [-1.0, -2.0, 0.0],  # class 2 is correct
], requires_grad=True)
target = torch.tensor([2, 2])

# Log softmax + NLL loss
log_probs = torch.nn.functional.log_softmax(logits, dim=1)
loss = torch.nn.functional.nll_loss(log_probs, target, reduction="mean")
loss.backward()

out = {
    "nll_loss": loss.tolist(),
    "logits_grad_shape": list(logits.grad.shape),
    "logits_grad_sum": logits.grad.sum().tolist(),
    "status": "OK"
}
print(json.dumps(out, indent=2))
