"""torch.save / torch.load zipfile format — roundtrip with real PyTorch."""

import base64
import io
import json
import torch


# ── 1. Build a small model, snapshot its state_dict ──────────
model = torch.nn.Sequential(
    torch.nn.Linear(3, 4),
    torch.nn.BatchNorm1d(4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 2),
)
sd = model.state_dict()


def _summarize(mapping):
    """Tolerate both real-PyTorch (Tensor values) and torch-pyodide (dict values)."""
    out = {}
    for k, v in mapping.items():
        if hasattr(v, "tolist"):
            out[k] = {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
        else:
            out[k] = {"shape": v["shape"], "dtype": v["dtype"]}
    return out


sd_summary = _summarize(sd)

# ── 2. Save to BytesIO and reload ─────────────────────────────
buf = io.BytesIO()
torch.save(sd, buf)
buf.seek(0)
loaded = torch.load(buf)
loaded_summary = _summarize(loaded)

# ── 3. Compute forward pass on original and on a clone loaded from the snapshot ──
x = torch.randn((2, 3))
y_before = model(x).tolist()

clone = torch.nn.Sequential(
    torch.nn.Linear(3, 4),
    torch.nn.BatchNorm1d(4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 2),
)
buf.seek(0)
clone.load_state_dict(torch.load(buf))
y_after = clone(x).tolist()
y_match = y_before == y_after

# ── 4. Inspect the archive layout ─────────────────────────────
import zipfile
buf.seek(0)
zf = zipfile.ZipFile(buf)
archive_names = sorted(zf.namelist())

# ── 5. Save to a file path and reload (covers the str-file API) ──
import tempfile, os
tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
tmp.close()
torch.save(sd, tmp.name)
loaded_from_file = torch.load(tmp.name)
file_load_ok = all(
    k in loaded_from_file and _summarize(loaded_from_file)[k]["shape"] == sd_summary[k]["shape"]
    for k in sd_summary
)
os.unlink(tmp.name)

out = {
    "sd_summary": sd_summary,
    "loaded_summary": loaded_summary,
    "y_match": y_match,
    "archive_names": archive_names,
    "file_load_ok": file_load_ok,
}
print(json.dumps(out, indent=2))
