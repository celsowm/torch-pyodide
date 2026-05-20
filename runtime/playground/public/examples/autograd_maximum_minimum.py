import torch

# Maximum + Minimum autograd
a = torch.tensor([1.0, 5.0, 3.0, 2.0], requires_grad=True)
b = torch.tensor([4.0, 2.0, 6.0, 1.0], requires_grad=True)

max_out = torch.maximum(a, b)
min_out = torch.minimum(a, b)

print("maximum:", max_out.tolist())
print("minimum:", min_out.tolist())

loss = max_out.sum() + min_out.sum()
loss.backward()
print("grad_a:", a.grad.tolist())
print("grad_b:", b.grad.tolist())
