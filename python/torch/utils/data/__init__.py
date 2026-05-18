from __future__ import annotations

from typing import Sequence, Iterator
from torch import Tensor


class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> object:
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors: Tensor) -> None:
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor")
        self.tensors = tensors
        self._len = tensors[0]._shape[0] if tensors[0]._shape else 0

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        return tuple(t[idx] for t in self.tensors)


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[object]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            import random
            random.shuffle(indices)
        batch: list[object] = []
        for idx in indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch:
            yield self._collate(batch)

    def _collate(self, batch: list[object]) -> object:
        if isinstance(batch[0], tuple):
            from torch import stack
            return tuple(stack([b[i] for b in batch]) for i in range(len(batch[0])))
        from torch import stack
        return stack(batch)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
