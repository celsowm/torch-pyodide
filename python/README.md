# torch-pyodide

**PyTorch-compatible API that runs entirely in the browser.**

Built on [Pyodide](https://pyodide.org) and [WebGPU](https://www.w3.org/TR/webgpu/), `torch-pyodide` lets you write PyTorch code that executes on the GPU inside the browser — no server, no CUDA, no installation beyond `pip install torch-pyodide`.

## What works

- **Tensor creation**: `tensor()`, `zeros`, `ones`, `rand`, `randn`, `arange`, `full`, `empty`
- **Arithmetic**: `add`, `sub`, `mul`, `div`, `pow`, `matmul`, `mm`, `mv`, `bmm`
- **Linear algebra**: `dot`, `outer`, `norm` (Frobenius, L1, L2, inf)
- **Unary ops**: `relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `sqrt`, `exp`, `log`, `sin`, `cos`, and 40+ more
- **Comparison**: `eq`, `ne`, `gt`, `lt`, `ge`, `le`
- **Reductions**: `sum`, `mean`, `max`, `min`, `prod`, `any`, `all`, `cumsum`, `cumprod`
- **Shape ops**: `reshape`, `flatten`, `squeeze`, `unsqueeze`, `transpose`, `permute`, `cat`, `stack`, `expand`
- **Indexing**: `select`, `slice`, `index_select`, `masked_select`, `masked_fill`, `where`
- **Neural network** (`torch.nn`): `Linear`, `Bilinear`, `Conv2d`, `BatchNorm1d/2d`, `LayerNorm`, `Dropout`, pooling, loss functions, activations
- **CUDA stub**: `torch.cuda.is_available()`, `torch.cuda.device_count()`, etc.

## Quick start

```python
import torch

x = torch.randn((3, 4))
w = torch.randn((4, 5))
y = x.matmul(w)
print(y.shape)  # (3, 5)

# Neural networks
model = torch.nn.Sequential(
    torch.nn.Linear(4, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 2),
)
logits = model(x)
```

## License

MIT
