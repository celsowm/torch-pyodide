"""Parity smoke for the newly added optimizers.

Each optimizer minimizes ``L = sum(w**2)`` (grad = 2w) for 5 deterministic
steps from a fixed initial tensor. Final parameters are compared against
reference values produced by real PyTorch 2.11. SparseAdam is compared to
Adam because, with a fully dense (all non-zero) gradient, the masked update
is mathematically the Adam update.
"""
import json

import torch

max_diff = 0.0
worst = ""
per = {}


def check(name, values, expected):
    global max_diff, worst
    local = 0.0
    for a, b in zip(values, expected):
        d = abs(a - b)
        local = max(local, d)
        if d > max_diff:
            max_diff = d
            worst = name
    per[name] = round(local, 6)


def run(make_opt, init, steps=5):
    w = torch.tensor(init, requires_grad=True)
    opt = make_opt([w])
    for _ in range(steps):
        loss = (w ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return [float(v) for v in w.detach().reshape(-1).tolist()]


VEC = [0.5, -1.0, 2.0, 0.25]
MAT = [[0.5, -1.0, 2.0], [1.5, -0.5, 0.75]]

check("ASGD", run(lambda p: torch.optim.ASGD(p, lr=0.1), VEC),
      [0.16383279860019684, -0.3276655972003937, 0.6553311944007874, 0.08191639930009842])
check("Adadelta", run(lambda p: torch.optim.Adadelta(p, lr=1.0), VEC),
      [0.48369595408439636, -0.9836313724517822, 1.9835991859436035, 0.23382633924484253])
check("Rprop", run(lambda p: torch.optim.Rprop(p, lr=0.1), VEC),
      [-0.03680002689361572, -0.2558399438858032, 1.2558398246765137, -0.04200000315904617])
check("Adafactor", run(lambda p: torch.optim.Adafactor(p, lr=0.1), VEC),
      [0.11226189136505127, -0.5491034984588623, 1.5236016511917114, 0.0010847174562513828])
check("SparseAdam", run(lambda p: torch.optim.SparseAdam(p, lr=0.1), VEC),
      [0.027814455330371857, -0.5079637169837952, 1.5029557943344116, -0.1276039481163025])
check("Muon", run(lambda p: torch.optim.Muon(p, lr=0.1), MAT),
      [0.5953269600868225, -0.7866916060447693, 1.5327621698379517,
       0.9495752453804016, -0.45031997561454773, 0.7565963864326477])

ok = max_diff < 5e-2
print(json.dumps({"ok": ok, "max_diff": max_diff, "worst": worst, "per": per}, sort_keys=True))
assert ok, f"max_diff {max_diff} worst {worst} per {per}"
