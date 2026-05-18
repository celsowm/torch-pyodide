"""
Garante que importar submódulos (torch.nn, torch.nn.functional) NÃO
disponibiliza torch.randn / torch.tensor / etc no namespace global.

Isto é para evitar o bug de alguém fazer:
    import torch.nn.functional as F
    logits = torch.randn((2, 3))  # Isto DEVE falhar

O PyTorch real também falha neste caso.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_py(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_nn_functional_does_not_leak_torch():
    code = """
import sys
sys.path.insert(0, '.')
import torch.nn.functional as F
try:
    torch.randn(3)
    print("LEAKED", end="")
except NameError:
    print("OK", end="")
"""
    result = _run_py(code)
    assert (
        "OK" in result.stdout
    ), f"Esperado NameError, mas torch.randn funcionou: {result.stdout}"


def test_nn_does_not_leak_torch():
    code = """
import sys
sys.path.insert(0, '.')
import torch.nn as nn
try:
    torch.zeros(3)
    print("LEAKED", end="")
except NameError:
    print("OK", end="")
"""
    result = _run_py(code)
    assert (
        "OK" in result.stdout
    ), f"Esperado NameError, mas torch.zeros funcionou: {result.stdout}"


def test_nn_func_modules_does_not_leak_torch_via_functional():
    code = """
import sys
sys.path.insert(0, '.')
from torch.nn.functional import relu
try:
    torch.ones(3)
    print("LEAKED", end="")
except NameError:
    print("OK", end="")
"""
    result = _run_py(code)
    assert (
        "OK" in result.stdout
    ), f"Esperado NameError, mas torch.ones funcionou: {result.stdout}"


if __name__ == "__main__":
    test_nn_functional_does_not_leak_torch()
    test_nn_does_not_leak_torch()
    test_nn_func_modules_does_not_leak_torch_via_functional()
    print("Todos os testes de regressao passaram.")
