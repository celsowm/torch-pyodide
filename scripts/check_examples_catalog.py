"""Verify every example referenced in examples.json exists on disk."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
catalog = json.loads((ROOT / "runtime" / "playground" / "public" / "examples.json").read_text(encoding="utf-8"))
examples_dir = ROOT / "runtime" / "playground" / "public" / "examples"
on_disk = {p.name for p in examples_dir.glob("*.py")}

missing = []
for entry in catalog["examples"]:
    fname = Path(entry["file"]).name
    if fname not in on_disk:
        missing.append(entry)

unused = sorted(on_disk - {Path(e["file"]).name for e in catalog["examples"]})

print(f"catalog: {len(catalog['examples'])} entries")
print(f"on disk: {len(on_disk)} .py files")
if missing:
    print("MISSING from disk:")
    for m in missing:
        print(f"  - {m['file']}")
if unused:
    print("UNUSED on disk (not in catalog):")
    for u in unused:
        print(f"  - {u}")
if not missing and not unused:
    print("OK: catalog and disk match 1:1")
