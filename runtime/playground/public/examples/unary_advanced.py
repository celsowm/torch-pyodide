import json
import torch
import torch.nn.functional as F

x = torch.tensor([[-1.0, 0.5, 1.0], [2.0, -2.0, 0.5]])
out = {
"input": x.tolist(),
"sigmoid": torch.sigmoid(x).tolist(),
"tanh": torch.tanh(x).tolist(),
"sin": torch.sin(x).tolist(),
"cos": torch.cos(x).tolist(),
"gelu": F.gelu(x).tolist(),
"silu": F.silu(x).tolist(),
"leaky_relu": F.leaky_relu(x, 0.01).tolist(),
"floor": torch.floor(x).tolist(),
"ceil": torch.ceil(x).tolist(),
"round": torch.round(x).tolist(),
"reciprocal": torch.reciprocal(x).tolist(),
"square": torch.square(x).tolist(),
"abs": torch.abs(x).tolist(),
"neg": torch.neg(x).tolist(),
"sqrt": torch.sqrt(x.abs()).tolist(),
"exp": torch.exp(x).tolist(),
"log": torch.log(x.abs() + 1.0).tolist(),
"sign": torch.sign(x).tolist(),
"softplus": F.softplus(x).tolist(),
"mish": F.mish(x).tolist(),
"hardswish": F.hardswish(x).tolist(),
"hardsigmoid": F.hardsigmoid(x).tolist(),
"softsign": F.softsign(x).tolist(),
"tanhshrink": F.tanhshrink(x).tolist(),
}
print(json.dumps(out, indent=2))
