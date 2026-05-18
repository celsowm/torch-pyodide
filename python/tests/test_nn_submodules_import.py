"""
Garante que todos os submódulos conhecidos de torch.nn importam sem
circular import / ImportError.

Executa em subprocess fora do Pyodide para testar a estrutura do
pacote Python independentemente do runtime.
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


_SUBMODULES = [
    "torch.nn",
    "torch.nn.functional",
    "torch.nn.modules",
    "torch.nn.init",
]


def test_all_nn_submodules_import():
    for mod in _SUBMODULES:
        code = f"""
import sys
sys.path.insert(0, '.')
import {mod}
print("OK", end="")
"""
        result = _run_py(code)
        assert "OK" in result.stdout, (
            f"Falha ao importar {mod}: {result.stderr.strip()}"
        )


if __name__ == "__main__":
    test_all_nn_submodules_import()
    print("Todos os submódulos importam sem erro.")
