# torch-pyodide

**PyTorch-compatible API that runs entirely in the browser.**
Built on [Pyodide](https://pyodide.org) and [WebGPU](https://www.w3.org/TR/webgpu/).

## Try it now

[Open the playground](https://celsowm.github.io/torch-pyodide/playground/) — write and run PyTorch code in your browser.

## Install

### In Pyodide (browser)

```python
import micropip
await micropip.install("torch-pyodide")
import torch

x = torch.randn((3, 4))
w = torch.randn((4, 5))
y = x.matmul(w)
print(y.shape)  # (3, 5)
```

### Locally (with Python + Node.js)

```bash
pip install torch-pyodide
# Requires Node.js 20+ and a WebGPU-capable browser/device
```

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

## License

MIT
