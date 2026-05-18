"""
Valida que todos os exemplos do playground compilam e executam
com o PyTorch real (instalado via pip).

Uso:
    python tests/validate_playground_examples.py

Requer: pip install torch
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = (
    ROOT / ".." / "runtime" / "playground" / "public" / "examples"
).resolve()
TORCH_PYODIDE_DIR = ROOT / "torch"


def _patch_example_for_real_torch(code: str) -> str:
    """Substitui imports de torch-pyodide por torch real e patches non-determinism."""
    # Remove imports de torch internos (nn.functional é o mesmo nome)
    code = code.replace("import torch.nn.functional as F", "import torch.nn.functional as F")
    # Já usa "import torch" — funciona com PyTorch real
    # Adiciona seed fixa após imports
    lines = code.split("\n")
    result: list[str] = []
    for line in lines:
        result.append(line)
        if line.startswith("import ") or line.startswith("from "):
            continue
        # After first non-import line, inject seed
        result.append('torch.manual_seed(42)')
        break
    # Insert seed at the top
    # Instead, prepend seed before execution
    return "\n".join(result)


def _exec_example(code: str, ns: dict[str, Any]) -> dict[str, Any]:
    """Executa o código do exemplo e retorna o JSON parseado do stdout."""
    import io
    import torch

    # Set deterministic seed
    torch.manual_seed(42)

    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        exec(code, ns)
    finally:
        sys.stdout = old_stdout

    output = buf.getvalue()
    if not output.strip():
        return {}

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"_raw_stdout": output.strip()}


def validate_example(filepath: Path) -> tuple[bool, str]:
    """Valida um exemplo do playground contra PyTorch real."""
    code = filepath.read_text(encoding="utf-8")

    ns: dict[str, Any] = {}
    error = None
    try:
        # Inject torch.manual_seed before example code
        patched = f"import torch\ntorch.manual_seed(42)\n{code}"
        exec(patched, ns)
    except Exception as e:
        error = str(e)

    if error:
        return False, error
    return True, "OK"


def main() -> int:
    errors = 0
    total = 0
    skipped = 0

    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch real não instalado. A saltar validação dos exemplos.")
        return 0

    example_files = sorted(EXAMPLES_DIR.glob("*.py"))
    print(f"A validar {len(example_files)} exemplos do playground contra PyTorch real...")
    print()

    for fp in example_files:
        total += 1
        ok, msg = validate_example(fp)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {fp.name}: {msg}")
        if not ok:
            errors += 1

    print()
    print(f"Resultado: {total - errors - skipped}/{total} OK", end="")
    if errors:
        print(f", {errors} erro(s)", end="")
    if skipped:
        print(f", {skipped} skip(s)", end="")
    print()

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
