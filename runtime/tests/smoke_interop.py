"""Smoke test the interop helper end-to-end (real-PyTorch only).

Generates a .pt archive via the example, pipes its base64 into the helper,
and confirms the helper loads it with matching keys and reproduces the
expected forward-pass output.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "playground" / "public" / "examples" / "save_load_interop.py"
HELPER = REPO_ROOT / "tests" / "interop_load_helper.py"


def main() -> int:
    gen = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        check=True,
        capture_output=True,
        text=True,
    )
    browser_payload = json.loads(gen.stdout)

    interop_input = json.dumps({
        "b64": browser_payload["b64"],
        "expected_keys": browser_payload["loaded_keys"],
    })
    helper = subprocess.run(
        [sys.executable, str(HELPER)],
        input=interop_input,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(helper.stdout)

    print(f"  browser loaded_keys:    {browser_payload['loaded_keys']}")
    print(f"  real-torch loaded_keys: {result['loaded_keys']}")
    print(f"  browser y_clone:        {browser_payload['y_clone']}")
    print(f"  real-torch y:           {result['y']}")
    print(f"  keys_match: {result['keys_match']}")

    if not result["keys_match"]:
        print("FAIL: key mismatch")
        return 1
    if result["y"] != browser_payload["y_clone"]:
        print("FAIL: forward pass mismatch")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
