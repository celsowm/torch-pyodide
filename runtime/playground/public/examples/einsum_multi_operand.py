import json
import torch

# Einsum with 3+ operands (iterative pairwise contraction)
a = torch.randn((2, 3))
b = torch.randn((3, 4))
c = torch.randn((4, 5))

# Three-matrix chain multiplication: a @ b @ c
result3 = torch.einsum('ij,jk,kl->il', a, b, c)

# Verify with manual matmul
manual = a.matmul(b).matmul(c)

# Four-matrix chain
d = torch.randn((5, 2))
result4 = torch.einsum('ij,jk,kl,lm->im', a, b, c, d)
manual4 = a.matmul(b).matmul(c).matmul(d)

# Batched matrix multiply with einsum
x = torch.randn((2, 3, 4))
y = torch.randn((2, 4, 5))
batch_result = torch.einsum('bij,bjk->bik', x, y)

out = {
    "three_matrix_shape": list(result3.shape),
    "three_matrix_manual_match": result3.tolist() == manual.tolist(),
    "four_matrix_shape": list(result4.shape),
    "four_matrix_manual_match": result4.tolist() == manual4.tolist(),
    "batched_shape": list(batch_result.shape),
    "status": "OK"
}
print(json.dumps(out, indent=2))
