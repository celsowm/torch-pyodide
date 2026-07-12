from __future__ import annotations

import inspect

import torch.optim as optim

NEW_OPTIM_API = [
    "ASGD",
    "Adadelta",
    "Adafactor",
    "Muon",
    "Rprop",
    "SparseAdam",
]


def test_optim_new_api_present():
    for name in NEW_OPTIM_API:
        assert hasattr(optim, name), f"torch.optim.{name} is missing"


def test_optim_new_api_constructor_params_match_installed_torch():
    import subprocess
    import sys
    import tempfile
    import json

    script = r"""
import json, inspect
import torch.optim as optim
names = json.loads(input())
out = {}
for name in names:
    sig = inspect.signature(getattr(optim, name).__init__)
    out[name] = [p for p in sig.parameters if p not in ("self", "params")]
print(json.dumps(out))
"""
    env = dict(__import__("os").environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tempfile.gettempdir(),
        env=env,
        input=json.dumps(NEW_OPTIM_API),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        import pytest

        pytest.skip(f"installed PyTorch not importable: {proc.stderr.strip()}")

    upstream = json.loads(proc.stdout)
    # Params implemented only in full PyTorch (perf/impl toggles) are optional.
    optional = {
        "foreach", "maximize", "differentiable", "capturable",
        "fused", "decoupled_weight_decay",
    }
    for name in NEW_OPTIM_API:
        local = [
            p for p in inspect.signature(getattr(optim, name).__init__).parameters
            if p not in ("self", "params")
        ]
        missing = [p for p in upstream[name] if p not in local and p not in optional]
        assert not missing, f"torch.optim.{name} missing constructor params {missing}"
