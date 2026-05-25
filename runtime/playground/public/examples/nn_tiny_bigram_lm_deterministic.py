import json

import torch
import torch.nn.functional as F
from torch import nn


text = "I like pytorch . I like AI ."
tokens = text.split()
vocab = sorted(set(tokens))
stoi = {token: i for i, token in enumerate(vocab)}
itos = {i: token for token, i in stoi.items()}

idx = torch.tensor([[stoi[token] for token in tokens]], dtype=torch.long)
x = idx[:, :-1]
y = idx[:, 1:]


class TinyDeterministicBigramLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_dim = 3
        self.wte = nn.Parameter(torch.tensor([
            [0.25, -0.10, 0.05],
            [-0.15, 0.30, 0.20],
            [0.40, 0.05, -0.25],
            [0.10, -0.35, 0.30],
            [-0.20, 0.15, -0.05],
        ], dtype=torch.float32))
        self.lm_weight = nn.Parameter(torch.tensor([
            [0.20, -0.15, 0.10],
            [-0.10, 0.25, 0.05],
            [0.15, 0.05, -0.20],
            [0.05, -0.10, 0.30],
            [-0.25, 0.10, 0.15],
        ], dtype=torch.float32))
        self.lm_bias = nn.Parameter(torch.tensor([0.02, -0.01, 0.03, -0.04, 0.01], dtype=torch.float32))

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        flat_ids = token_ids.reshape(-1)
        token_vectors = torch.index_select(self.wte, 0, flat_ids)
        return token_vectors.reshape(token_ids.shape[0], token_ids.shape[1], self.embedding_dim)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        token_vectors = self.embed(token_ids)
        logits = F.linear(token_vectors, self.lm_weight, self.lm_bias)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(token_ids)
            last_position = logits.shape[1] - 1
            next_logits = logits[:, last_position, :]
            next_id = next_logits.argmax(dim=-1, keepdim=True)
            token_ids = torch.cat([token_ids, next_id], dim=1)
        return token_ids


model = TinyDeterministicBigramLM()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)

_, loss_before = model(x, y)
for _ in range(12):
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

logits_after, loss_after = model(x, y)
prompt = torch.tensor([[stoi["I"]]], dtype=torch.long)
generated = model.generate(prompt, max_new_tokens=6)
generated_ids = [int(i) for i in generated[0].tolist()]
generated_text = " ".join(itos[i] for i in generated_ids)

print(json.dumps({
    "loss_before": round(float(loss_before.item()), 6),
    "loss_after": round(float(loss_after.item()), 6),
    "generated_ids": generated_ids,
    "generated_text": generated_text,
    "last_logits": [round(float(v), 6) for v in logits_after[0, logits_after.shape[1] - 1, :].tolist()],
}, sort_keys=True))
