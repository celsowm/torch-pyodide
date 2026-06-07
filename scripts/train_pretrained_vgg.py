"""Train a MiniVGG in real PyTorch and export a JSON bundle (state_dict +
sample input + reference predictions) that gets embedded directly into the
playground example at
    runtime/playground/public/examples/real_model_pretrained_vgg.py

The MiniVGG is a small VGG-like CNN with:
  * 3 conv blocks, each Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
  * A 2-layer classifier with Dropout(0.5)
  * Trained with Adam + StepLR (lr halves every 10 epochs)

The point of this example is to exercise BatchNorm2d (running_mean and
running_var in the state_dict), Dropout (no-op in eval mode), and
StepLR (used during training) — all in a single end-to-end "load and
infer" flow. The browser sees the trained state_dict, reconstructs
MiniVGG, calls `.eval()` (which switches BN to running stats and
disables Dropout), and runs inference. Predictions must match real
PyTorch bit-for-bit.

Run from the repo root:
    python scripts/train_pretrained_vgg.py
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


# ── Architecture (must match the example) ───────────────────────
class MiniVGG(nn.Module):
    """A small VGG-like CNN for 3x16x16 RGB inputs, 5 output classes.

    Three conv blocks (each: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d)
    followed by a 2-layer classifier with Dropout. ~9.5k parameters,
    ~38 KB serialized, ~50 KB after base64.

    The state_dict carries 8 weight tensors, 4 bias tensors, 3 BN
    weight/bias pairs, 3 BN running_mean, 3 BN running_var, and 3 BN
    num_batches_tracked scalars.
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.5) -> None:
        super().__init__()
        # Block 1: 3 -> 16 channels, 16x16 -> 8x8
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # Block 2: 16 -> 32 channels, 8x8 -> 4x4
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # Block 3: 32 -> 64 channels, 4x4 -> 2x2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # Classifier
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ── Training on synthetic data ──────────────────────────────────
def train(
    seed: int = 42,
    epochs: int = 30,
    batch: int = 16,
    lr: float = 1e-3,
    step_size: int = 10,
    gamma: float = 0.5,
) -> tuple[MiniVGG, torch.Tensor, dict]:
    torch.manual_seed(seed)
    model = MiniVGG()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(opt, step_size=step_size, gamma=gamma)

    # Synthetic 3x16x16 inputs, 5-class labels. Deterministic.
    n_train = 64
    x = torch.randn(n_train, 3, 16, 16)
    y = torch.randint(0, 5, (n_train,))

    history = {"loss": [], "lr": []}
    for _ in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            xb, yb = x[idx], y[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        history["loss"].append(epoch_loss / n_batches)
        history["lr"].append(opt.param_groups[0]["lr"])

    model.eval()
    return model, x[:2].detach().clone(), history


# ── Reference predictions (used by the parity test) ────────────
def reference_predictions(model: MiniVGG, x: torch.Tensor) -> dict:
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
EXAMPLE_PATH = REPO_ROOT / "runtime" / "playground" / "public" / "examples" / "real_model_pretrained_vgg.py"
JSON_PATH = REPO_ROOT / "runtime" / "playground" / "public" / "pretrained" / "vgg_mini.json"
TEMPLATE_PATH = REPO_ROOT / "scripts" / "real_model_pretrained_vgg.template.py"


def main() -> None:
    model, sample_x, history = train()
    sd_bytes = io.BytesIO()
    torch.save(model.state_dict(), sd_bytes)

    payload = {
        "version": 1,
        "state_dict_b64": base64.b64encode(sd_bytes.getvalue()).decode("ascii"),
        "sample_x": sample_x.tolist(),
        "predictions": reference_predictions(model, sample_x),
        "training": {
            "epochs": len(history["loss"]),
            "initial_loss": round(history["loss"][0], 6),
            "final_loss": round(history["loss"][-1], 6),
            "initial_lr": round(history["lr"][0], 8),
            "final_lr": round(history["lr"][-1], 8),
            "scheduler": "StepLR(step_size=10, gamma=0.5)",
        },
        "architecture": {
            "blocks": 3,
            "channels": [16, 32, 64],
            "has_batchnorm": True,
            "has_dropout": True,
            "dropout_p": 0.5,
        },
    }

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload))

    if not TEMPLATE_PATH.exists():
        print(f"WARNING: template not found at {TEMPLATE_PATH}")
        return

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    # `json.dumps(payload)` returns a *Python string literal* — double
    # quotes inside the payload are escaped, so the result can be
    # substituted directly into a Python source file.
    rendered = template.replace("__BUNDLE_JSON_PLACEHOLDER__", json.dumps(payload))
    if "__BUNDLE_JSON_PLACEHOLDER__" in rendered:
        raise RuntimeError("placeholder not found in template")
    EXAMPLE_PATH.write_text(rendered, encoding="utf-8")

    print(f"wrote {JSON_PATH} ({JSON_PATH.stat().st_size} bytes)")
    print(f"wrote {EXAMPLE_PATH} ({EXAMPLE_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
