from __future__ import annotations

import json
import subprocess
import sys
import tempfile

import pytest

import torch.fft as fft

NEW_FFT_API = [
    "hfft",
    "ihfft",
    "hfft2",
    "hfftn",
    "ihfft2",
    "ihfftn",
    "irfft2",
    "irfftn",
]


def test_fft_new_api_present():
    for name in NEW_FFT_API:
        assert hasattr(fft, name), f"torch.fft.{name} is missing"


def test_fft_name_parity_with_installed_torch():
    """Every public callable of the installed ``torch.fft`` exists in ours."""
    script = r"""
import json
import torch
names = sorted(
    n for n in dir(torch.fft)
    if not n.startswith("_") and callable(getattr(torch.fft, n))
)
print(json.dumps(names))
"""
    env = dict(__import__("os").environ)
    env.pop("PYTHONPATH", None)
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tempfile.gettempdir(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        pytest.skip(f"installed PyTorch is not importable: {process.stderr.strip()}")

    upstream = json.loads(process.stdout)
    missing = [n for n in upstream if not hasattr(fft, n)]
    assert not missing, f"torch.fft missing {missing} vs upstream"
