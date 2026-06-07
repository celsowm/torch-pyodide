"""
Fase 12.7: ResNet8 + HuberLoss + CosineAnnealingLR + ReduceLROnPlateau + LBFGS.

Exercises:
- ResNet8 (BasicBlock-based) architecture for 3x16x16 inputs.
- nn.HuberLoss and F.huber_loss (smooth L1 with delta transition).
- CosineAnnealingLR smoothly anneals the LR from base to eta_min.
- ReduceLROnPlateau reduces the LR when a metric stops improving.
- LBFGS optimizer fits a small regression problem to a low loss.
- End-to-end: ResNet8 trained with HuberLoss + CosineAnnealingLR.
"""
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
)


# ── ResNet8 architecture ─────────────────────────────────────────
class BasicBlock(nn.Module):
    """A standard ResNet BasicBlock: 2x (Conv2d -> BN -> ReLU) with a skip."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        # Downsample: 1x1 conv + BN when stride != 1 or channels change.
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return F.relu(out)


class ResNet8(nn.Module):
    """ResNet-8 for small (3x16x16) inputs: 1 conv + 1 BasicBlock per stage, 3 stages.

    Total weight layers: 1 + 3*2 + 1 (FC) = 8.
    """

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # Stages: 1 BasicBlock each, channels [16, 32, 64], strides [1, 2, 2].
        self.layer1 = self._make_layer(16, 1, stride=1)
        self.layer2 = self._make_layer(32, 1, stride=2)
        self.layer3 = self._make_layer(64, 1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_planes, planes, stride=stride)]
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)


def main() -> None:
    # ── 1. HuberLoss: forward + autograd through a small example ─────
    torch.manual_seed(0)
    pred = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    target = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
    huber_mod = nn.HuberLoss(delta=1.0)
    huber_val = huber_mod(pred, target)
    huber_func = F.huber_loss(pred, target, delta=1.0)
    huber_match = abs(huber_val.item() - huber_func.item()) < 1e-6

    # Test autograd: with reduction='mean' and |diff| < delta, the gradient
    # of `0.5 * diff^2` w.r.t. pred is `diff / N` where N = total elements.
    huber_val.sum().backward()
    huber_grad = pred.grad.tolist()
    N = 6  # 2*3 elements
    expected_grad = [[(1.0 - 1.5) / N, (2.0 - 2.5) / N, (3.0 - 3.5) / N],
                     [(4.0 - 4.5) / N, (5.0 - 5.5) / N, (6.0 - 6.5) / N]]
    huber_grad_match = all(
        abs(huber_grad[i][j] - expected_grad[i][j]) < 1e-5 for i in range(2) for j in range(3)
    )

    # ── 2. ResNet8: forward + backward + eval mode sanity ──────────
    torch.manual_seed(1)
    model = ResNet8(num_classes=5)
    x = torch.randn((4, 3, 16, 16))
    y_target = torch.randint(0, 5, (4,))
    model.train()
    y = model(x)
    forward_shape = list(y.shape)
    loss = F.cross_entropy(y, y_target)
    loss.backward()
    first_grad_norm = float(sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None) ** 0.5)

    # Eval mode: forward still works, no NaN, finite outputs.
    model.eval()
    with torch.no_grad():
        y_eval = model(x)
    eval_outputs_finite = all(
        math.isfinite(y_eval[i][j].item()) for i in range(4) for j in range(5)
    )
    eval_outputs_match_shape = list(y_eval.shape) == [4, 5]

    # ── 3. CosineAnnealingLR: smoothly anneal LR over T_max epochs ─
    torch.manual_seed(2)
    m_lin = nn.Linear(2, 1)
    opt_cos = torch.optim.SGD(m_lin.parameters(), lr=0.1)
    sch_cos = CosineAnnealingLR(opt_cos, T_max=10, eta_min=0.001)
    lrs: list[float] = []
    for _ in range(11):
        lrs.append(opt_cos.param_groups[0]["lr"])
        sch_cos.step()
    # At epoch 0 LR is base (0.1), at T_max it's eta_min.
    cos_lr_shape_ok = lrs[0] == 0.1 and abs(lrs[-1] - 0.001) < 1e-5
    # LR should be monotonically decreasing (cosine goes from 1 to 0 over [0, T_max]).
    cos_monotone = all(lrs[i] >= lrs[i + 1] - 1e-9 for i in range(len(lrs) - 1))
    cos_mid_value = lrs[5]  # halfway should be near midpoint

    # ── 4. ReduceLROnPlateau: reduce LR on plateau ────────────────
    torch.manual_seed(3)
    m_plat = nn.Linear(2, 1)
    opt_plat = torch.optim.SGD(m_plat.parameters(), lr=0.1)
    sch_plat = ReduceLROnPlateau(opt_plat, mode="min", factor=0.5, patience=2)
    plateau_metrics = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8]
    for m_val in plateau_metrics:
        sch_plat.step(m_val)
    plateau_lr_final = opt_plat.param_groups[0]["lr"]
    # After 5 non-improving steps (patience=2 triggers at step 3+), LR should be < 0.1.
    plateau_lr_reduced = plateau_lr_final < 0.1

    # ── 5. LBFGS: fit a small regression problem to a low loss ────
    torch.manual_seed(4)
    m_lbfgs = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    opt_lbfgs = torch.optim.LBFGS(m_lbfgs.parameters(), lr=1.0, max_iter=5, history_size=10)
    x_lb = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    y_lb = torch.tensor([[1.0], [1.0]])

    def closure() -> torch.Tensor:
        opt_lbfgs.zero_grad()
        out = m_lbfgs(x_lb)
        loss = ((out - y_lb) ** 2).mean()
        loss.backward()
        return loss

    lbfgs_initial = float(closure().item())
    losses_lbfgs: list[float] = []
    for _ in range(10):
        v = opt_lbfgs.step(closure)
        losses_lbfgs.append(float(v))
    lbfgs_final = losses_lbfgs[-1]
    lbfgs_converged = lbfgs_final < lbfgs_initial * 0.05

    # ── 6. End-to-end: ResNet8 trained with HuberLoss + CosineAnnealingLR
    torch.manual_seed(5)
    model_e2e = ResNet8(num_classes=5)
    opt_e2e = torch.optim.Adam(model_e2e.parameters(), lr=0.01)
    sch_e2e = CosineAnnealingLR(opt_e2e, T_max=20, eta_min=0.0001)
    crit = nn.HuberLoss(delta=1.0)

    # Synthetic regression target: predict the mean pixel value (continuous).
    n_samples = 32
    x_e2e = torch.randn((n_samples, 3, 16, 16))
    y_e2e_scalar = x_e2e.mean(dim=2)  # (n_samples, 3, 16)
    y_e2e_scalar = y_e2e_scalar.mean(dim=2)  # (n_samples, 3)
    y_e2e = y_e2e_scalar.mean(dim=1, keepdim=True) * 0.1  # (n_samples, 1)
    # Repeat to 5-dim so the head matches.
    y_e2e_5d = y_e2e.repeat(1, 5)

    losses_e2e: list[float] = []
    for _ in range(30):
        model_e2e.train()
        opt_e2e.zero_grad()
        pred = model_e2e(x_e2e)
        loss = crit(pred, y_e2e_5d)
        loss.backward()
        opt_e2e.step()
        sch_e2e.step()
        losses_e2e.append(float(loss.item()))

    e2e_loss_decreased = losses_e2e[-1] < losses_e2e[0] * 0.7

    out = {
        # HuberLoss
        "huber_value": round(float(huber_val.item()), 6),
        "huber_match_functional": bool(huber_match),
        "huber_grad_match": bool(huber_grad_match),
        # ResNet8
        "forward_shape": forward_shape,
        "first_grad_norm": round(first_grad_norm, 4),
        "eval_outputs_finite": bool(eval_outputs_finite),
        "eval_outputs_match_shape": bool(eval_outputs_match_shape),
        # CosineAnnealingLR
        "cos_lr_first": lrs[0],
        "cos_lr_last": round(lrs[-1], 6),
        "cos_lr_shape_ok": bool(cos_lr_shape_ok),
        "cos_monotone_decreasing": bool(cos_monotone),
        "cos_mid_value": round(cos_mid_value, 4),
        # ReduceLROnPlateau
        "plateau_lr_initial": 0.1,
        "plateau_lr_final": round(plateau_lr_final, 6),
        "plateau_lr_reduced": bool(plateau_lr_reduced),
        # LBFGS
        "lbfgs_initial_loss": round(lbfgs_initial, 4),
        "lbfgs_final_loss": round(lbfgs_final, 6),
        "lbfgs_converged": bool(lbfgs_converged),
        "lbfgs_losses_first_3": [round(v, 4) for v in losses_lbfgs[:3]],
        # End-to-end ResNet8 training
        "e2e_losses_first_5": [round(v, 4) for v in losses_e2e[:5]],
        "e2e_losses_last_5": [round(v, 4) for v in losses_e2e[-5:]],
        "e2e_loss_decreased": bool(e2e_loss_decreased),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
