from __future__ import annotations

from typing import Sequence
from torch import Tensor


def pad_sequence(sequences: Sequence[Tensor], batch_first: bool = False, padding_value: float = 0.0) -> Tensor:
    import torch
    max_len = max(s._shape[0] for s in sequences)
    padded: list[Tensor] = []
    for seq in sequences:
        seq_len = seq._shape[0]
        if seq_len < max_len:
            pad_shape = [max_len - seq_len] + list(seq._shape[1:])
            pad_tensor = torch.full(pad_shape, padding_value, dtype=seq.dtype)
            padded.append(torch.cat([seq, pad_tensor], dim=0))
        else:
            padded.append(seq)
    result = torch.stack(padded, dim=0 if batch_first else 1)
    return result


# ── Gradient Clipping ────────────────────────────────────────────

def clip_grad_norm_(parameters: Sequence[Tensor], max_norm: float, norm_type: float = 2.0) -> float:
    """Clips gradient norm of iterable of parameters. Modifies gradients in-place."""
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return 0.0
    if norm_type == float("inf"):
        total_norm = max(p.grad.abs().max().item() for p in parameters)
    else:
        total_norm = sum((p.grad ** norm_type).sum() for p in parameters)
        total_norm = total_norm ** (1.0 / norm_type)
    total_norm = float(total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            p.grad = p.grad.mul(clip_coef)
    return total_norm


def clip_grad_value_(parameters: Sequence[Tensor], clip_value: float) -> None:
    """Clips gradient values to [-clip_value, clip_value]. Modifies gradients in-place."""
    parameters = [p for p in parameters if p.grad is not None]
    for p in parameters:
        p.grad = p.grad.clamp(-clip_value, clip_value)
