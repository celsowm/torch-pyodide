from __future__ import annotations

from typing import Sequence, Iterator, Callable, Optional
from torch import Tensor
import random


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


# ── Samplers ──────────────────────────────────────────────────────

class Sampler:
    """Base class for all Samplers."""
    def __init__(self, data_source: Dataset) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data_source)


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order."""
    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))


class RandomSampler(Sampler):
    """Samples elements randomly without replacement."""
    def __init__(self, data_source: Dataset, replacement: bool = False, num_samples: Optional[int] = None) -> None:
        super().__init__(data_source)
        self.replacement = replacement
        self._num_samples = num_samples if num_samples is not None else len(data_source)

    def __iter__(self) -> Iterator[int]:
        if self.replacement:
            return iter(random.choices(range(len(self.data_source)), k=self._num_samples))
        indices = list(range(len(self.data_source)))
        random.shuffle(indices)
        return iter(indices[:self._num_samples])


class BatchSampler(Sampler):
    """Wraps another sampler to yield batches of indices."""
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False) -> None:
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices."""
    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        indices_list = list(self.indices)
        random.shuffle(indices_list)
        return iter(indices_list)

    def __len__(self) -> int:
        return len(self.indices)


class WeightedRandomSampler(Sampler):
    """Samples elements from [0, len(weights)-1] with given probabilities."""
    def __init__(self, weights: Sequence[float], num_samples: int, replacement: bool = False) -> None:
        self.weights = list(weights)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self) -> Iterator[int]:
        return iter(random.choices(range(len(self.weights)), weights=self.weights, k=self.num_samples))

    def __len__(self) -> int:
        return self.num_samples


# ── Dataset compositions ──────────────────────────────────────────

class ConcatDataset(Dataset):
    """Concatenates multiple datasets."""
    def __init__(self, datasets: Sequence[Dataset]) -> None:
        self.datasets = list(datasets)
        self.cumulative_sizes = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> object:
        import bisect
        ds_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if ds_idx == 0:
            return self.datasets[0][idx]
        return self.datasets[ds_idx][idx - self.cumulative_sizes[ds_idx - 1]]


class Subset(Dataset):
    """Subset of a dataset at specified indices."""
    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> object:
        return self.dataset[self.indices[idx]]


# ── Collate functions ─────────────────────────────────────────────

def default_collate(batch: list[object]) -> object:
    """Default collate function: stacks tensors."""
    elem = batch[0]
    if isinstance(elem, Tensor):
        from torch import stack
        return stack(batch)
    elif isinstance(elem, tuple):
        return tuple(default_collate([b[i] for b in batch]) for i in range(len(elem)))
    elif isinstance(elem, list):
        return [default_collate([b[i] for b in batch]) for i in range(len(elem))]
    elif isinstance(elem, (int, float)):
        from torch import tensor
        return tensor(batch)
    else:
        return batch


def default_convert(data: object) -> object:
    """Default convert function: returns data as-is."""
    return data


# ── DataLoader ────────────────────────────────────────────────────

class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        worker_init_fn: Optional[Callable] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate
        self.pin_memory = pin_memory
        self.worker_init_fn = worker_init_fn

        if batch_sampler is not None:
            self.batch_sampler = batch_sampler
        elif sampler is not None:
            self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        else:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
            self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)

    def __iter__(self) -> Iterator[object]:
        if self.worker_init_fn is not None:
            self.worker_init_fn(0)
        for batch_indices in self.batch_sampler:
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(batch)

    def __len__(self) -> int:
        return len(self.batch_sampler)
