import json
import torch

# DataLoader with Samplers
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, BatchSampler, ConcatDataset, Subset, default_collate

# Create two datasets
ds1 = TensorDataset(torch.tensor([[1.0], [2.0], [3.0]]), torch.tensor([[10.0], [20.0], [30.0]]))
ds2 = TensorDataset(torch.tensor([[4.0], [5.0]]), torch.tensor([[40.0], [50.0]]))

# Concatenate datasets
concat_ds = ConcatDataset([ds1, ds2])
print(f"ConcatDataset length: {len(concat_ds)}")

# DataLoader with custom sampler
dl = DataLoader(concat_ds, batch_size=2, shuffle=True, drop_last=False)

for i, batch in enumerate(dl):
    print(f"Batch {i}: x={batch[0].tolist()}, y={batch[1].tolist()}")

# Subset
sub = Subset(concat_ds, [0, 2, 4])
dl_sub = DataLoader(sub, batch_size=1)
print(f"Subset length: {len(sub)}")
for batch in dl_sub:
    print(f"Subset batch: {batch[0].tolist()}")

out = {"concat_len": len(concat_ds), "sub_len": len(sub), "status": "OK"}
print(json.dumps(out, indent=2))
