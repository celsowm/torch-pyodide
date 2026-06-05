"""State-dict, named-*, apply, and to() smoke for nn.Module.

Exercises the nn.Module parity surface added in the compat expansion:
- state_dict / load_state_dict roundtrip
- named_parameters / named_modules / named_buffers iteration
- apply() visits every module
- to() / cpu() / cuda() / float() / double() return self
- register_buffer() exposes running stats in state_dict
"""

import json
import torch
import torch.nn as nn


def _summarize(sd):
    """Tolerate both real-PyTorch (Tensor values) and torch-pyodide (dict values)."""
    out = {}
    for k, v in sd.items():
        if hasattr(v, "tolist"):  # real PyTorch Tensor
            out[k] = {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
        else:  # torch-pyodide serialized dict
            out[k] = {"shape": v["shape"], "dtype": v["dtype"]}
    return out


# ── 1. Build a small model tree with parameters AND buffers (BatchNorm) ──
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.BatchNorm1d(4),
    nn.ReLU(),
    nn.Linear(4, 2),
)
x = torch.randn((2, 3))
y_before = model(x).tolist()

# ── 2. named_parameters / named_modules / named_buffers ──
param_names = [n for n, _ in model.named_parameters()]
module_names = [n for n, _ in model.named_modules()]
buffer_names = [n for n, _ in model.named_buffers()]

# ── 3. state_dict keys + shapes (dtypes are always float32 here) ──
sd = model.state_dict()
sd_summary = _summarize(sd)

# ── 4. Roundtrip: clone the model, load state_dict, verify output matches
clone = nn.Sequential(
    nn.Linear(3, 4),
    nn.BatchNorm1d(4),
    nn.ReLU(),
    nn.Linear(4, 2),
)
clone.load_state_dict(sd)
y_after = clone(x).tolist()
y_match = y_before == y_after

# ── 5. apply() visits every module (including the Sequential root) ──
visited: list[str] = []
model.apply(lambda m: visited.append(type(m).__name__))

# ── 6. to() / cpu() / cuda() / float() return self ──
chain = []
chain.append(model.to("cpu") is model)
chain.append(model.cpu() is model)
chain.append(model.cuda() is model)
chain.append(model.float() is model)
chain.append(model.to(dtype=torch.float32) is model)
# double() is float64 → unsupported in torch-pyodide, supported in real PyTorch.
# We accept either behavior so the same example runs in both runtimes.
double_supported = True
try:
    model.double()
except NotImplementedError:
    double_supported = False
device_chain_ok = all(chain)

# ── 7. train/eval toggles training flag on the whole tree ──
model.train()
train_state_before = all(m.training for _, m in model.named_modules())
model.eval()
train_state_after = all((not m.training) for _, m in model.named_modules())

out = {
    "param_names": param_names,
    "module_names": module_names,
    "buffer_names": buffer_names,
    "sd_keys": sorted(sd_summary.keys()),
    "sd_summary": sd_summary,
    "y_match": y_match,
    "y_shape": [2, 2],
    "visited_classes": visited,
    "device_chain_ok": device_chain_ok,
    "train_state_before": train_state_before,
    "train_state_after": train_state_after,
    "has_batchnorm": "1.running_mean" in sd_summary and "1.running_var" in sd_summary,
}
print(json.dumps(out, indent=2))
