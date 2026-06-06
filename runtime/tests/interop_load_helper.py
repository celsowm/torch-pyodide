"""Helper: decode a base64 .pt archive, load it as a state_dict, and run
forward on a Sequential matching the example's architecture. Used by the
browser test to verify that the file produced by torch-pyodide can be loaded
by an external real-PyTorch runtime.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import torch


def main() -> None:
    payload = json.loads(sys.stdin.read())
    b64 = payload["b64"]
    expected_keys = sorted(payload["expected_keys"])

    raw = base64.b64decode(b64)
    buf = io.BytesIO(raw)
    loaded = torch.load(buf, weights_only=False)

    # Build a clone Sequential with the same shape as the example.
    clone = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )
    clone.load_state_dict(loaded)

    # Re-run forward on the same input the example used (deterministic).
    x = torch.tensor([[0.5, -0.3, 1.0], [-0.8, 0.2, 0.9]])
    y = clone(x).tolist()

    print(json.dumps({
        "loaded_keys": sorted(loaded.keys()),
        "expected_keys": expected_keys,
        "keys_match": sorted(loaded.keys()) == expected_keys,
        "y": [[round(v, 4) for v in row] for row in y],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
