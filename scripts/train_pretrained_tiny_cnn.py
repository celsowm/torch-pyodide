"""Train a tiny CNN in real PyTorch and emit a JSON bundle (state_dict +
sample input + reference predictions) that gets embedded directly into the
playground example at
    runtime/playground/public/examples/real_model_pretrained_tiny_cnn.py

The same bundle is also written to
    runtime/playground/public/pretrained/tiny_cnn.json
so it can be inspected or fetched by hand.

Run from the repo root:
    python scripts/train_pretrained_tiny_cnn.py
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Architecture (must match the example) ───────────────────────────────────
class TinyCNN(nn.Module):
    """A 2-conv + 2-fc CNN for 3x16x16 RGB inputs, 5 output classes.

    ~3.4k parameters, ~14 KB serialized (small enough to embed in the
    playground example as a base64 literal).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)   # 3*8*3*3 + 8 = 224
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 8*16*3*3 + 16 = 1168
        self.fc1 = nn.Linear(16 * 4 * 4, 32)                    # 16*4*4*32 + 32 = 8224
        self.fc2 = nn.Linear(32, 5)                             # 32*5 + 5 = 165

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ── Training on synthetic data ──────────────────────────────────────────────
def train(seed: int = 42, steps: int = 80, batch: int = 16) -> tuple[TinyCNN, torch.Tensor]:
    torch.manual_seed(seed)
    model = TinyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Synthetic 3x16x16 inputs, 5-class labels. Deterministic.
    x = torch.randn(batch, 3, 16, 16)
    y = torch.randint(0, 5, (batch,))

    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

    model.eval()
    return model, x[:2].detach().clone()


# ── Reference predictions (used by the parity test) ────────────────────────
def reference_predictions(model: TinyCNN, x: torch.Tensor) -> dict[str, list[list[float]]]:
    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)
    return {
        "logits": logits.tolist(),
        "probs": probs.tolist(),
        "preds": preds.tolist(),
    }


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PATH = REPO_ROOT / "runtime" / "playground" / "public" / "examples" / "real_model_pretrained_tiny_cnn.py"
JSON_PATH = REPO_ROOT / "runtime" / "playground" / "public" / "pretrained" / "tiny_cnn.json"
EXAMPLE_TEMPLATE_PATH = REPO_ROOT / "scripts" / "real_model_pretrained_tiny_cnn.template.py"


def main() -> None:
    model, sample_x = train()
    sd_bytes = io.BytesIO()
    torch.save(model.state_dict(), sd_bytes)

    # Report the final training loss/accuracy on the 16 training examples.
    # The model has overfit these (it was trained for 80 steps on the same
    # fixed batch), so loss should be ~0 and accuracy should be 1.0 — this
    # is what tells the user the state_dict is the trained one, not random
    # initial weights.
    torch.manual_seed(42)
    model_init = TinyCNN()  # consume the same RNG state the model consumed
    train_x = torch.randn(16, 3, 16, 16)
    train_y = torch.randint(0, 5, (16,))
    del model_init
    with torch.no_grad():
        train_logits = model(train_x)
        train_loss = F.cross_entropy(train_logits, train_y).item()
        train_acc = (train_logits.argmax(dim=-1) == train_y).float().mean().item()

    payload = {
        "version": 1,
        "state_dict_b64": base64.b64encode(sd_bytes.getvalue()).decode("ascii"),
        "sample_x": sample_x.tolist(),
        "predictions": reference_predictions(model, sample_x),
        "training": {
            "final_loss": round(train_loss, 6),
            "final_accuracy": round(train_acc, 4),
            "n_train_samples": 16,
            "n_train_steps": 80,
        },
    }

    # 1. Write the standalone JSON bundle (for inspection / external fetch).
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload))

    # 2. Regenerate the example file from a checked-in template by replacing
    #    the `_BUNDLE_JSON` placeholder with the actual JSON literal.
    if not EXAMPLE_TEMPLATE_PATH.exists():
        print(f"WARNING: template not found at {EXAMPLE_TEMPLATE_PATH}; "
              f"leaving {EXAMPLE_PATH} untouched.")
        print(f"wrote {JSON_PATH} ({JSON_PATH.stat().st_size} bytes)")
        return

    template = EXAMPLE_TEMPLATE_PATH.read_text(encoding="utf-8")
    # `json.dumps(payload)` returns a *Python string literal* — double quotes
    # inside the payload are escaped, so the result can be substituted
    # directly into a Python source file.
    rendered = template.replace("__BUNDLE_JSON_PLACEHOLDER__", json.dumps(payload))
    if "__BUNDLE_JSON_PLACEHOLDER__" in rendered:
        raise RuntimeError("placeholder not found in template")
    EXAMPLE_PATH.write_text(rendered, encoding="utf-8")

    print(f"wrote {JSON_PATH} ({JSON_PATH.stat().st_size} bytes)")
    print(f"wrote {EXAMPLE_PATH} ({EXAMPLE_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
