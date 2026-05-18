import json, os, datetime, importlib

with open(os.path.join(os.path.dirname(__file__), "registry.json")) as f:
    registry = json.load(f)

targets = registry["targets"]
statuses = []

for t in targets:
    tid = t["id"]
    kind = t.get("kind", "")
    implemented = False

    try:
        if kind == "module_func":
            mod_name = tid.split(".", 1)[1] if tid.startswith("torch.") else None
            import torch
            if mod_name and hasattr(torch, mod_name):
                implemented = True

        elif kind == "nn_func":
            parts = tid.split(".")
            func_name = parts[-1]
            from torch.nn import functional as F
            if hasattr(F, func_name):
                implemented = True

        elif kind == "nn_class":
            parts = tid.split(".")
            cls_name = parts[-1]
            import torch.nn as nn
            if hasattr(nn, cls_name):
                implemented = True

        elif kind in ("tensor_method", "tensor_property"):
            attr = tid.split(".")[-1]
            from torch._tensor import Tensor
            if hasattr(Tensor, attr):
                implemented = True

        elif kind == "cuda_func":
            parts = tid.split(".")
            import torch
            obj = torch.cuda
            for p in parts[2:]:
                obj = getattr(obj, p)
            implemented = True
    except Exception:
        pass

    statuses.append({"id": tid, "kind": kind, "implemented": implemented})

total = len(statuses)
done = sum(1 for s in statuses if s["implemented"])
pct = done / total * 100 if total else 0

print("Progress: %d/%d (%.2f%%)" % (done, total, pct))

cats = {}
for s in statuses:
    cat = s["kind"].split("_")[0] if "_" in s["kind"] else s["kind"]
    cats.setdefault(cat, {"total": 0, "done": 0})
    cats[cat]["total"] += 1
    if s["implemented"]:
        cats[cat]["done"] += 1

for cat, v in sorted(cats.items()):
    print("  %s: %d/%d" % (cat, v["done"], v["total"]))

unimpl = [s for s in statuses if not s["implemented"]]
if unimpl:
    print("\nUnimplemented:")
    for s in unimpl:
        print("  - %s (%s)" % (s["id"], s["kind"]))

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
report = {
    "generated_at": now,
    "total_targets": total,
    "implemented": done,
    "coverage_pct": round(pct, 2),
    "targets": statuses,
}
out_path = os.path.join(os.path.dirname(__file__), "report.json")
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print("\nReport written to compatibility/report.json")
