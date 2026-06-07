import torch
import torch.nn as nn
import torch.nn.functional as F
import json

torch.manual_seed(42)

vocab_size = 20
embed_dim = 8
num_classes = 5
batch_size = 4
seq_len = 6
lr = 0.05
num_steps = 15

embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
head = nn.Linear(embed_dim, num_classes)
optimizer = torch.optim.Adam(list(embed.parameters()) + list(head.parameters()), lr=lr)

losses = []
for step in range(num_steps):
    idx = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    targets = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

    h = embed(idx)
    pooled = h.mean(dim=1)
    logits = head(pooled)
    loss = F.cross_entropy(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

w_grad = embed.weight.grad
grad_norm = w_grad.norm().item() if w_grad is not None else -1.0

emb_0 = embed.weight[0].tolist()
emb_1 = embed.weight[1].tolist()

result = {
    "embed_shape": list(embed.weight.shape),
    "forward_shape": list(h.shape),
    "pooled_shape": list(pooled.shape),
    "logits_shape": list(logits.shape),
    "loss_start": losses[0],
    "loss_end": losses[-1],
    "loss_decreased": losses[-1] < losses[0],
    "grad_norm": grad_norm,
    "padding_idx_zero": all(v == 0.0 for v in emb_0),
    "status": "OK",
}

print(json.dumps(result))
