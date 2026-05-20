import torch

# Sort + Topk autograd
x = torch.tensor([3.0, 1.0, 2.0, 5.0, 4.0], requires_grad=True)

values_sorted, indices = x.sort()
print("sorted values:", values_sorted.tolist())
print("sorted indices:", indices.tolist())

values_topk, indices_topk = x.topk(3)
print("topk values:", values_topk.tolist())
print("topk indices:", indices_topk.tolist())

# backward
loss = values_topk.sum()
loss.backward()
print("grad:", x.grad.tolist())
