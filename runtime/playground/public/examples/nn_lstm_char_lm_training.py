import json
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)

text = "the quick brown fox jumps over the lazy dog . the quick brown fox runs ."
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}

ids = torch.tensor([[stoi[ch] for ch in text]], dtype=torch.long)
x = ids[:, :-1]
y = ids[:, 1:]
T = x.shape[1]


class CharLSTM(nn.Module):

    def __init__(self, vocab_size: int, n_embd: int, hidden_size: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(input_size=n_embd, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x_emb = self.emb(idx)
        out, _ = self.lstm(x_emb)
        logits = self.head(out)
        return logits


model = CharLSTM(vocab_size=vocab_size, n_embd=16, hidden_size=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

logits_before = model(x)
B, T_, V = logits_before.shape
loss_before = F.cross_entropy(logits_before.reshape(B * T_, V), y.reshape(B * T_))

for step in range(10):
    logits = model(x)
    B, T_, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * T_, V), y.reshape(B * T_))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

logits_after = model(x)
B, T_, V = logits_after.shape
loss_after = F.cross_entropy(logits_after.reshape(B * T_, V), y.reshape(B * T_))

result = {
    "vocab": vocab_size,
    "T": T,
    "loss_before": float(loss_before.item()),
    "loss_after": float(loss_after.item()),
    "loss_decreased": float(loss_after.item()) < float(loss_before.item()),
    "hidden_size": model.lstm.hidden_size,
    "num_layers": model.lstm.num_layers,
    "state_dict_keys": len(model.state_dict()),
}
print(json.dumps(result))
