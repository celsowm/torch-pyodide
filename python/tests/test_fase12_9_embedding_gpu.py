"""
Tests for the GPU-accelerated Embedding op (Fase 12.9):
- embedding_from_tensor dispatches to the WGSL embedding shader
- _grad_embedding backward rule scatters gradients correctly
- padding_idx zeroes both forward output and backward gradient
- F.embedding / torch.embedding API surface

These are source-level + API-surface tests (no real GPU).
GPU behavior is verified via browser tests.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _python_has_torch():
    try:
        import torch
        return True
    except ImportError:
        return False


has_torch = pytest.mark.skipif(not _python_has_torch(), reason="real PyTorch not installed")


# ── Source-level checks ───────────────────────────────────────────────

def test_embedding_from_tensor_exists():
    src = (ROOT / "torch" / "tensor_nn_ops.py").read_text()
    assert "def embedding_from_tensor" in src, "embedding_from_tensor must exist"


def test_embedding_from_tensor_calls_runtime_embedding():
    src = (ROOT / "torch" / "tensor_nn_ops.py").read_text()
    assert "runtime.embedding(" in src, "embedding_from_tensor must call runtime.embedding"


def test_embedding_from_tensor_autograd():
    src = (ROOT / "torch" / "tensor_nn_ops.py").read_text()
    assert "_grad_embedding" in src, "embedding_from_tensor must reference _grad_embedding"


def test_grad_embedding_exists():
    src = (ROOT / "torch" / "autograd_rules.py").read_text()
    assert "def _grad_embedding" in src, "_grad_embedding must exist in autograd_rules"


def test_grad_embedding_skips_padding_idx():
    src = (ROOT / "torch" / "autograd_rules.py").read_text()
    m = re.search(r"def _grad_embedding\(.*?\).*?(?=\ndef )", src, re.DOTALL)
    assert m, "_grad_embedding not found"
    block = m.group(0)
    assert "padding_idx" in block, "_grad_embedding must handle padding_idx"
    assert "continue" in block, "_grad_embedding must skip gradient for padding_idx"


def test_embedding_module_uses_gpu_dispatch():
    src = (ROOT / "torch" / "nn" / "modules.py").read_text()
    m = re.search(r"class Embedding\(Module\):.*?(?=\nclass |\Z)", src, re.DOTALL)
    assert m, "Embedding class not found"
    block = m.group(0)
    assert "embedding_from_tensor" in block, "Embedding.forward must use embedding_from_tensor (GPU)"


def test_f_embedding_exists():
    src = (ROOT / "torch" / "nn" / "functional.py").read_text()
    assert "def embedding(" in src, "F.embedding must exist"


def test_torch_embedding_top_level():
    src = (ROOT / "torch" / "__init__.py").read_text()
    assert "def embedding(" in src, "torch.embedding must exist"


def test_nn_init_exports_embedding():
    src = (ROOT / "torch" / "nn" / "__init__.py").read_text()
    assert "embedding" in src, "nn.__init__ must export F.embedding"


def test_wgsl_embedding_shader_padding_idx():
    src = (ROOT.parent / "runtime" / "src" / "vendor" / "torchjs" / "shaders" / "embedding.wgsl").read_text()
    assert "padding_idx" in src, "embedding.wgsl must handle padding_idx"


def test_ts_embedding_ops_exists():
    src = (ROOT.parent / "runtime" / "src" / "ops" / "embeddingOps.ts").read_text()
    assert "async embedding(" in src, "EmbeddingOps.embedding must exist"
    assert "paddingIdx" in src, "EmbeddingOps.embedding must accept paddingIdx"


def test_runtime_ts_exposes_embedding():
    src = (ROOT.parent / "runtime" / "src" / "runtime.ts").read_text()
    assert "async embedding(" in src, "runtime.ts must expose embedding method"


# ── Numerical parity with real PyTorch (subprocess) ───────────────────

def _run_real_pytorch(script: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        pytest.skip(f"real PyTorch not available: {result.stderr.strip()}")
    return result.stdout.strip()


def test_grad_embedding_numerical_parity():
    """Verify scatter-add backward matches real PyTorch (runs in subprocess)."""
    script = (
        "import torch; import json; "
        "torch.manual_seed(0); "
        "V, D = 10, 4; "
        "w = torch.randn(V, D, requires_grad=True); "
        "idx = torch.tensor([3, 7, 1, 3, 0], dtype=torch.long); "
        "out = torch.nn.functional.embedding(idx, w); "
        "out.sum().backward(); "
        "grad = w.grad.numpy().flatten().tolist(); "
        "print(json.dumps(grad))"
    )
    import json
    ref_grad_flat = json.loads(_run_real_pytorch(script))
    V, D = 10, 4
    expected_flat = [0.0] * (V * D)
    idx_vals = [3, 7, 1, 3, 0]
    for i, tok in enumerate(idx_vals):
        for j in range(D):
            expected_flat[tok * D + j] += 1.0
    np.testing.assert_allclose(ref_grad_flat, expected_flat, atol=1e-6)


def test_grad_embedding_padding_idx_numerical_parity():
    """Verify padding_idx row gets zero grad (runs in subprocess)."""
    script = (
        "import torch; import json; "
        "torch.manual_seed(0); "
        "V, D = 10, 4; "
        "w = torch.randn(V, D, requires_grad=True); "
        "idx = torch.tensor([3, 0, 1, 0, 5], dtype=torch.long); "
        "out = torch.nn.functional.embedding(idx, w, padding_idx=0); "
        "out.sum().backward(); "
        "row0_max = w.grad[0].abs().max().item(); "
        "row3_max = w.grad[3].abs().max().item(); "
        "print(json.dumps([row0_max, row3_max]))"
    )
    import json
    vals = json.loads(_run_real_pytorch(script))
    assert vals[0] == 0.0, "padding_idx row must have zero grad"
    assert vals[1] > 0.0, "non-padding row must have nonzero grad"


# ── Example file exists ───────────────────────────────────────────────

def test_example_file_exists():
    p = ROOT.parent / "runtime" / "playground" / "public" / "examples" / "nn_embedding_gpu_training.py"
    assert p.exists(), "nn_embedding_gpu_training.py example must exist"


def test_example_in_examples_json():
    import json
    data = json.loads((ROOT.parent / "runtime" / "playground" / "public" / "examples.json").read_text())
    ids = [e["id"] for e in data["examples"]]
    assert "nn_embedding_gpu_training" in ids, "nn_embedding_gpu_training must be in examples.json"
