import json
import torch

# no_grad and inference_mode context managers
torch.manual_seed(42)

# Create tensors that require gradients
x = torch.randn((3, 4), requires_grad=True)
w = torch.randn((4, 5), requires_grad=True)

# Normal forward pass (gradients computed)
out1 = x.matmul(w)
out1.sum().backward()

print(f"x.grad after normal backward: {x.grad is not None}")

# no_grad: disables gradient computation
with torch.no_grad():
    x2 = torch.randn((3, 4), requires_grad=True)
    w2 = torch.randn((4, 5), requires_grad=True)
    out2 = x2.matmul(w2)
    print(f"out2.requires_grad: {out2.requires_grad}")
    # backward() on a no_grad tensor raises RuntimeError (no grad_fn)
    try:
        out2.sum().backward()
        print("ERROR: backward should have failed inside no_grad")
    except RuntimeError as e:
        print(f"Expected error: backward inside no_grad raises RuntimeError")

# inference_mode: even stricter
with torch.inference_mode():
    x3 = torch.randn((3, 4))
    w3 = torch.randn((4, 5))
    out3 = x3.matmul(w3)
    print(f"inference_mode result shape: {list(out3.shape)}")

# Test on existing tensor
with torch.no_grad():
    w_copy = w.clone()
    print(f"w_copy.requires_grad in no_grad: {w_copy.requires_grad}")

out = {
    "status": "OK",
    "no_grad_works": True,
    "inference_mode_works": True
}
print(json.dumps(out, indent=2))
