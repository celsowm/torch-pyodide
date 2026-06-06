"""torch.save / torch.load cross-runtime interop smoke.

The example produces a base64-encoded .pt archive (built from a small
sequential model) and a forward-pass output computed locally. The browser
test then feeds that base64 into a separate real-PyTorch subprocess to
verify that the file can be loaded by an external runtime, and vice versa.
"""
import base64
import io
import json

import torch


model = torch.nn.Sequential(
    torch.nn.Linear(3, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 2),
)
sd = model.state_dict()

# ── 1. Save to a base64-encoded .pt archive ───────────────────────────────
buf = io.BytesIO()
torch.save(sd, buf)
b64 = base64.b64encode(buf.getvalue()).decode("ascii")

# ── 2. Forward pass on the original model ─────────────────────────────────
x = torch.tensor([[0.5, -0.3, 1.0], [-0.8, 0.2, 0.9]])
y_original = model(x).tolist()

# ── 3. Self roundtrip: decode + load + forward must match ─────────────────
buf2 = io.BytesIO(base64.b64decode(b64))
loaded = torch.load(buf2)

clone = torch.nn.Sequential(
    torch.nn.Linear(3, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 2),
)
clone.load_state_dict(loaded)
y_clone = clone(x).tolist()
y_match = y_original == y_clone

print(json.dumps({
    "b64": b64,
    "b64_length": len(b64),
    "loaded_keys": sorted(loaded.keys()),
    "y_original": [[round(v, 4) for v in row] for row in y_original],
    "y_clone": [[round(v, 4) for v in row] for row in y_clone],
    "y_match": y_match,
}, sort_keys=True))
