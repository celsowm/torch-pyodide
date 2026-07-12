from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import torch.linalg as linalg

ROOT = Path(__file__).resolve().parents[1]

NEW_LINALG_API = [
    "LinAlgError",
    "cholesky_ex",
    "inv_ex",
    "lu_factor_ex",
    "solve_ex",
    "ldl_factor",
    "ldl_factor_ex",
    "ldl_solve",
    "householder_product",
    "vecdot",
    "tensorinv",
    "tensorsolve",
]


def test_linalg_new_api_present():
    for name in NEW_LINALG_API:
        assert hasattr(linalg, name), f"torch.linalg.{name} is missing"


def test_linalg_new_api_signatures_match_installed_torch():
    """Compare signatures of the new ``torch.linalg`` functions vs installed torch."""
    script = r"""
import json
import torch

names = json.loads(input())
signatures = {}
for name in names:
    obj = getattr(torch.linalg, name, None)
    doc = getattr(obj, "__doc__", "") or ""
    signature = ""
    for line in doc.splitlines():
        candidate = line.strip().replace("\\*", "*")
        if candidate.startswith(f"{name}(") or candidate.startswith(f"torch.linalg.{name}("):
            signature = candidate
            break
    signatures[name] = signature
print(json.dumps(signatures))
"""
    env = dict(__import__("os").environ)
    env.pop("PYTHONPATH", None)
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tempfile.gettempdir(),
        env=env,
        input=json.dumps(NEW_LINALG_API),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        pytest.skip(f"installed PyTorch is not importable: {process.stderr.strip()}")

    import inspect

    upstream = json.loads(process.stdout)
    ignored = {"out", "device", "layout", "dtype", "memory_format", "generator", "requires_grad"}
    for name in NEW_LINALG_API:
        upstream_sig = upstream.get(name, "")
        if not upstream_sig:
            continue
        # crude param-name extraction between the first '(' and '->' / ')'
        params_blob = upstream_sig[upstream_sig.find("(") + 1 :]
        end = params_blob.find(") ->")
        params_blob = params_blob[:end] if end >= 0 else params_blob.rstrip(")")
        upstream_params = {
            p.split(":", 1)[0].split("=", 1)[0].strip().lstrip("*")
            for p in params_blob.split(",")
            if p.strip() and p.strip() != "*"
        }
        local_params = set(inspect.signature(getattr(linalg, name)).parameters)
        missing = upstream_params - local_params - {"self"} - ignored
        assert not missing, f"torch.linalg.{name} missing params {missing} vs upstream"
