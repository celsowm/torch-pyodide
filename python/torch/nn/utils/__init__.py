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
