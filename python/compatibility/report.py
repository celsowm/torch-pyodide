from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "compatibility" / "registry.json"
REPORT_JSON_PATH = ROOT / "compatibility" / "report.json"
REPORT_MD_PATH = ROOT / "compatibility" / "report.md"


@dataclass
class TargetResult:
    id: str
    kind: str
    implemented: bool
    note: str


def _check_target(torch_mod: Any, tensor_cls: Any, target: dict[str, str]) -> TargetResult:
    target_id = target["id"]
    kind = target["kind"]

    if kind == "module_func":
        name = target_id.split(".", 1)[1]
        ok = hasattr(torch_mod, name) and callable(getattr(torch_mod, name))
        return TargetResult(target_id, kind, ok, "ok" if ok else "missing")

    if kind == "tensor_method":
        name = target_id.split(".", 1)[1]
        ok = hasattr(tensor_cls, name) and callable(getattr(tensor_cls, name))
        return TargetResult(target_id, kind, ok, "ok" if ok else "missing")

    if kind == "tensor_property":
        name = target_id.split(".", 1)[1]
        attr = getattr(tensor_cls, name, None)
        ok = isinstance(attr, property) or attr is not None
        return TargetResult(target_id, kind, ok, "ok" if ok else "missing")

    if kind == "cuda_func":
        parts = target_id.split(".")[1:]  # drop torch
        current = torch_mod
        for part in parts:
            if not hasattr(current, part):
                return TargetResult(target_id, kind, False, "missing")
            current = getattr(current, part)
        ok = callable(current)
        return TargetResult(target_id, kind, ok, "ok" if ok else "missing")

    return TargetResult(target_id, kind, False, f"unknown kind: {kind}")


def main() -> None:
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    sys.path.insert(0, str(ROOT))
    import torch as local_torch

    targets = registry["targets"]
    results = [_check_target(local_torch, local_torch.Tensor, t) for t in targets]
    implemented = sum(1 for r in results if r.implemented)
    total = len(results)
    pct = (implemented / total * 100.0) if total else 0.0

    by_kind: dict[str, dict[str, int]] = {}
    for r in results:
        group = by_kind.setdefault(r.kind, {"implemented": 0, "total": 0})
        group["total"] += 1
        if r.implemented:
            group["implemented"] += 1

    report_json = {
        "implemented": implemented,
        "total": total,
        "percentage": round(pct, 2),
        "by_kind": by_kind,
        "targets": [r.__dict__ for r in results],
    }

    REPORT_JSON_PATH.write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    lines = [
        "# torch-pyodide compatibility report",
        "",
        f"- Implemented: **{implemented}/{total}**",
        f"- Coverage: **{pct:.2f}%**",
        "",
        "## By kind",
    ]

    for kind, data in sorted(by_kind.items()):
        kpct = (data["implemented"] / data["total"] * 100.0) if data["total"] else 0.0
        lines.append(f"- `{kind}`: {data['implemented']}/{data['total']} ({kpct:.2f}%)")

    lines.append("")
    lines.append("## Missing targets")
    missing = [r for r in results if not r.implemented]
    if not missing:
        lines.append("- none")
    else:
        for r in missing:
            lines.append(f"- `{r.id}` ({r.kind})")

    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Compatibility: {implemented}/{total} ({pct:.2f}%)")
    print(f"- JSON: {REPORT_JSON_PATH}")
    print(f"- Markdown: {REPORT_MD_PATH}")


if __name__ == "__main__":
    main()
