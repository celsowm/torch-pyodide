from __future__ import annotations

import builtins as _builtins
from typing import Sequence

from ._version import __version__

# Import autograd FIRST - must be before any other imports that might trigger Torch runtime
from .autograd import (
    no_grad,
    inference_mode,
    set_grad_enabled,
    is_grad_enabled,
    grad,
)

from . import cuda
from ._api_creation import (
    arange,
    bernoulli,
    empty,
    empty_like,
    exponential,
    eye,
    full,
    full_like,
    linspace,
    log_normal,
    logspace,
    manual_seed,
    multinomial,
    normal,
    ones,
    ones_like,
    rand,
    randint,
    randn,
    randperm,
    seed,
    tensor,
    zeros,
    zeros_like,
)
from ._einsum_impl import einsum
from ._runtime import _get_runtime
from ._tensor import Tensor
from .tensor_factories_ops import (
    arange_from_values,
    bernoulli_from_shape,
    empty_from_shape,
    empty_like_from_tensor,
    exponential_from_shape,
    full_from_shape,
    full_like_from_tensor,
    log_normal_from_shape,
    normal_from_shape,
    ones_from_shape,
    ones_like_from_tensor,
    rand_from_shape,
    randn_from_shape,
    tensor_from_data,
    zeros_from_shape,
    zeros_like_from_tensor,
)
from .tensor_ops import (
    cat_from_tensors,
    expand_from_tensor,
    index_select_from_tensor,
    stack_from_tensors,
    where_from_tensors,
    sigmoid_from_tensor,
    tanh_from_tensor,
    sin_from_tensor,
    cos_from_tensor,
    gelu_from_tensor,
    silu_from_tensor,
    leaky_relu_from_tensor,
    floor_from_tensor,
    ceil_from_tensor,
    round_from_tensor,
    reciprocal_from_tensor,
    square_from_tensor,
    eq_from_tensors,
    ne_from_tensors,
    lt_from_tensors,
    le_from_tensors,
    gt_from_tensors,
    ge_from_tensors,
    sum_dim_from_tensor,
    mean_dim_from_tensor,
    prod_from_tensor,
    min_from_tensor,
    max_from_tensor,
    masked_select_from_tensor,
    masked_fill_from_tensor,
    nonzero_from_tensor,
    roll_from_tensor,
    equal_from_tensors,
    allclose_from_tensors,
    isclose_from_tensors,
    log_softmax_from_tensor,
    softmax_from_tensor,
    # New in this session
    tan_from_tensor,
    asin_from_tensor,
    acos_from_tensor,
    atan_from_tensor,
    sinh_from_tensor,
    cosh_from_tensor,
    asinh_from_tensor,
    acosh_from_tensor,
    atanh_from_tensor,
    exp2_from_tensor,
    log2_from_tensor,
    log10_from_tensor,
    log1p_from_tensor,
    expm1_from_tensor,
    trunc_from_tensor,
    frac_from_tensor,
    softplus_from_tensor,
    mish_from_tensor,
    hardsigmoid_from_tensor,
    hardswish_from_tensor,
    softsign_from_tensor,
    tanhshrink_from_tensor,
    rsqrt_from_tensor,
    sign_from_tensor,
    sgn_from_tensor,
    isnan_from_tensor,
    isinf_from_tensor,
    isfinite_from_tensor,
    isposinf_from_tensor,
    isneginf_from_tensor,
    logical_not_from_tensor,
    erf_from_tensor,
    erfc_from_tensor,
    lgamma_from_tensor,
    digamma_from_tensor,
    i0_from_tensor,
    deg2rad_from_tensor,
    rad2deg_from_tensor,
    pow_from_tensors,
    heaviside_from_tensors,
    maximum_from_tensors,
    minimum_from_tensors,
    any_from_tensor,
    all_from_tensor,
    cumsum_from_tensor,
    cumprod_from_tensor,
    tril_from_tensor,
    triu_from_tensor,
    flip_from_tensor,
    topk_from_tensor,
    sort_from_tensor,
    gather_from_tensor,
    scatter_from_tensor,
    atan2_from_tensors,
    hypot_from_tensors,
    logaddexp_from_tensors,
    logaddexp2_from_tensors,
    fmod_from_tensors,
    remainder_from_tensors,
    xlogy_from_tensors,
    copysign_from_tensors,
    floor_divide_from_tensors,
    true_divide_from_tensors,
    nextafter_from_tensors,
    logical_and_from_tensors,
    logical_or_from_tensors,
    logical_xor_from_tensors,
    bitwise_and_from_tensors,
    bitwise_or_from_tensors,
    bitwise_xor_from_tensors,
    bitwise_not_from_tensor,
    lerp_from_tensors,
    addcmul_from_tensors,
    addcdiv_from_tensors,
    mul_scalar_from_tensor,
)

__all__ = [
    "Tensor",
    "cuda",
    "tensor",
    "zeros",
    "ones",
    "rand",
    "randn",
    "arange",
    "full",
    "full_like",
    "zeros_like",
    "ones_like",
    "empty",
    "empty_like",
    "eye",
    "randint",
    "randperm",
    "multinomial",
    "linspace",
    "logspace",
    "normal",
    "bernoulli",
    "exponential",
    "log_normal",
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "matmul",
    "atan2",
    "hypot",
    "fmod",
    "remainder",
    "fmax",
    "fmin",
    "logaddexp",
    "logaddexp2",
    "lerp",
    "addcmul",
    "addcdiv",
    "xlogy",
    "copysign",
    "nextafter",
    "floor_divide",
    "true_divide",
    "logical_and",
    "logical_or",
    "logical_xor",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    "relu",
    "abs",
    "sqrt",
    "exp",
    "log",
    "neg",
    "clamp",
    "where",
    "argmax",
    "argmin",
    "reshape",
    "flatten",
    "squeeze",
    "unsqueeze",
    "transpose",
    "permute",
    "select",
    "slice",
    "cat",
    "stack",
    "expand",
    "index_select",
    "sigmoid",
    "tanh",
    "sin",
    "cos",
    "gelu",
    "silu",
    "leaky_relu",
    "floor",
    "ceil",
    "round",
    "reciprocal",
    "square",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "asinh",
    "acosh",
    "atanh",
    "exp2",
    "log2",
    "log10",
    "log1p",
    "expm1",
    "trunc",
    "frac",
    "softplus",
    "mish",
    "hardsigmoid",
    "hardswish",
    "softsign",
    "tanhshrink",
    "rsqrt",
    "sign",
    "sgn",
    "isnan",
    "isinf",
    "isfinite",
    "isposinf",
    "isneginf",
    "logical_not",
    "erf",
    "erfc",
    "lgamma",
    "digamma",
    "i0",
    "deg2rad",
    "rad2deg",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "maximum",
    "minimum",
    "heaviside",
    "sum",
    "mean",
    "prod",
    "min",
    "max",
    "any",
    "all",
    "cumsum",
    "cumprod",
    "masked_select",
    "masked_fill",
    "nonzero",
    "roll",
    "equal",
    "allclose",
    "isclose",
    "softmax",
    "log_softmax",
    "tril",
    "triu",
    "flip",
    "mm",
    "bmm",
    "mv",
    "dot",
    "outer",
    "norm",
    "split",
    "chunk",
    "repeat",
    "tile",
    "no_grad",
    "inference_mode",
    "set_grad_enabled",
    "is_grad_enabled",
    "topk",
    "sort",
    "gather",
    "scatter",
    "einsum",
    "save",
    "load",
    "diag",
    "distributions",
    "linalg",
    # dtype constants / aliases
    "float32",
    "float16",
    "bfloat16",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool_",
    "long",
    "bool",
    "double",
    "half",
    "int",
    "float",
    "short",
    "char",
    "byte",
]


def add(a: Tensor, b: Tensor) -> Tensor:
    return a.add(b)


def sub(a: Tensor, b: Tensor) -> Tensor:
    return a.sub(b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    return a.mul(b)


def div(a: Tensor, b: Tensor) -> Tensor:
    return a.div(b)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a.matmul(b)


def mm(a: Tensor, b: Tensor) -> Tensor:
    return a.mm(b)


def bmm(a: Tensor, b: Tensor) -> Tensor:
    return a.bmm(b)


def mv(a: Tensor, b: Tensor) -> Tensor:
    return a.mv(b)


def dot(a: Tensor, b: Tensor) -> Tensor:
    return a.dot(b)


def outer(a: Tensor, b: Tensor) -> Tensor:
    return a.outer(b)


def norm(x: Tensor, p: float | str = "fro") -> Tensor:
    return x.norm(p)


def relu(x: Tensor) -> Tensor:
    return x.relu()


def abs(x: Tensor) -> Tensor:
    return x.abs()


def sqrt(x: Tensor) -> Tensor:
    return x.sqrt()


def exp(x: Tensor) -> Tensor:
    return x.exp()


def log(x: Tensor) -> Tensor:
    return x.log()


def neg(x: Tensor) -> Tensor:
    return x.neg()


def clamp(x: Tensor, min: float | None = None, max: float | None = None) -> Tensor:
    return x.clamp(min=min, max=max)


def where(condition: Tensor, x: Tensor | float | int | bool, y: Tensor | float | int | bool) -> Tensor:
    if not isinstance(x, Tensor):
        dtype = y.dtype if isinstance(y, Tensor) else "float32"
        x = full_like(condition, _builtins.float(x), dtype=dtype)
    if not isinstance(y, Tensor):
        dtype = x.dtype if isinstance(x, Tensor) else "float32"
        y = full_like(condition, _builtins.float(y), dtype=dtype)
    return where_from_tensors(condition, x, y)


def argmax(x: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    return x.argmax(dim=dim, keepdim=keepdim)


def argmin(x: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    return x.argmin(dim=dim, keepdim=keepdim)


def reshape(input: Tensor, shape: int | Sequence[int]) -> Tensor:
    return input.reshape(shape)


def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    return input.flatten(start_dim=start_dim, end_dim=end_dim)


def squeeze(input: Tensor, dim: int | None = None) -> Tensor:
    return input.squeeze(dim=dim)


def unsqueeze(input: Tensor, dim: int) -> Tensor:
    return input.unsqueeze(dim=dim)


def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    return input.transpose(dim0=dim0, dim1=dim1)


def permute(input: Tensor, dims: Sequence[int]) -> Tensor:
    return input.permute(dims=dims)


def select(input: Tensor, dim: int, index: int) -> Tensor:
    return input.select(dim=dim, index=index)


def slice(input: Tensor, dim: int, start: int | None = None, end: int | None = None, step: int = 1) -> Tensor:
    return input.slice(dim=dim, start=start, end=end, step=step)


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    # Pairwise cat for >2 tensors (runtime only supports 2)
    ts = list(tensors)
    while len(ts) > 2:
        new_ts: list[Tensor] = []
        for i in range(0, len(ts), 2):
            if i + 1 < len(ts):
                new_ts.append(cat_from_tensors([ts[i], ts[i + 1]], dim=dim))
            else:
                new_ts.append(ts[i])
        ts = new_ts
    if len(ts) == 2:
        return cat_from_tensors([ts[0], ts[1]], dim=dim)
    return ts[0]


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    return stack_from_tensors(tensors, dim=dim)


def expand(input: Tensor, shape: int | Sequence[int]) -> Tensor:
    return expand_from_tensor(input, shape=shape)


def index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
    return index_select_from_tensor(input, dim=dim, index=index)


def embedding(input: Tensor, weight: Tensor, padding_idx: int = -1) -> Tensor:
    from .tensor_nn_ops import embedding_from_tensor
    return embedding_from_tensor(weight, input, padding_idx)


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def sin(x: Tensor) -> Tensor:
    return x.sin()


def cos(x: Tensor) -> Tensor:
    return x.cos()


def gelu(x: Tensor) -> Tensor:
    return x.gelu()


def silu(x: Tensor) -> Tensor:
    return x.silu()


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    return x.leaky_relu(negative_slope)


def floor(x: Tensor) -> Tensor:
    return x.floor()


def ceil(x: Tensor) -> Tensor:
    return x.ceil()


def round(x: Tensor) -> Tensor:
    return x.round()


def reciprocal(x: Tensor) -> Tensor:
    return x.reciprocal()


def square(x: Tensor) -> Tensor:
    return x.square()


def tan(x: Tensor) -> Tensor:
    return x.tan()


def asin(x: Tensor) -> Tensor:
    return x.asin()


def acos(x: Tensor) -> Tensor:
    return x.acos()


def atan(x: Tensor) -> Tensor:
    return x.atan()


def sinh(x: Tensor) -> Tensor:
    return x.sinh()


def cosh(x: Tensor) -> Tensor:
    return x.cosh()


def asinh(x: Tensor) -> Tensor:
    return x.asinh()


def acosh(x: Tensor) -> Tensor:
    return x.acosh()


def atanh(x: Tensor) -> Tensor:
    return x.atanh()


def exp2(x: Tensor) -> Tensor:
    return x.exp2()


def log2(x: Tensor) -> Tensor:
    return x.log2()


def log10(x: Tensor) -> Tensor:
    return x.log10()


def log1p(x: Tensor) -> Tensor:
    return x.log1p()


def expm1(x: Tensor) -> Tensor:
    return x.expm1()


def trunc(x: Tensor) -> Tensor:
    return x.trunc()


def frac(x: Tensor) -> Tensor:
    return x.frac()


def softplus(x: Tensor) -> Tensor:
    return x.softplus()


def mish(x: Tensor) -> Tensor:
    return x.mish()


def hardsigmoid(x: Tensor) -> Tensor:
    return x.hardsigmoid()


def hardswish(x: Tensor) -> Tensor:
    return x.hardswish()


def softsign(x: Tensor) -> Tensor:
    return x.softsign()


def tanhshrink(x: Tensor) -> Tensor:
    return x.tanhshrink()


def rsqrt(x: Tensor) -> Tensor:
    return x.rsqrt()


def sign(x: Tensor) -> Tensor:
    return x.sign()


def sgn(x: Tensor) -> Tensor:
    return x.sgn()


def isnan(x: Tensor) -> Tensor:
    return x.isnan()


def isinf(x: Tensor) -> Tensor:
    return x.isinf()


def isfinite(x: Tensor) -> Tensor:
    return x.isfinite()


def isposinf(x: Tensor) -> Tensor:
    return x.isposinf()


def isneginf(x: Tensor) -> Tensor:
    return x.isneginf()


def logical_not(x: Tensor) -> Tensor:
    return x.logical_not()


def erf(x: Tensor) -> Tensor:
    return x.erf()


def erfc(x: Tensor) -> Tensor:
    return x.erfc()


def lgamma(x: Tensor) -> Tensor:
    return x.lgamma()


def digamma(x: Tensor) -> Tensor:
    return x.digamma()


def i0(x: Tensor) -> Tensor:
    return x.i0()


def deg2rad(x: Tensor) -> Tensor:
    return x.deg2rad()


def rad2deg(x: Tensor) -> Tensor:
    return x.rad2deg()


def pow(a: Tensor, b: Tensor) -> Tensor:
    return pow_from_tensors(a, b)


def atan2(a: Tensor, b: Tensor) -> Tensor:
    return atan2_from_tensors(a, b)


def hypot(a: Tensor, b: Tensor) -> Tensor:
    return hypot_from_tensors(a, b)


def fmod(a: Tensor, b: Tensor) -> Tensor:
    return fmod_from_tensors(a, b)


def remainder(a: Tensor, b: Tensor) -> Tensor:
    return remainder_from_tensors(a, b)


def fmax(a: Tensor, b: Tensor) -> Tensor:
    return maximum_from_tensors(a, b)


def fmin(a: Tensor, b: Tensor) -> Tensor:
    return minimum_from_tensors(a, b)


def logaddexp(a: Tensor, b: Tensor) -> Tensor:
    return logaddexp_from_tensors(a, b)


def logaddexp2(a: Tensor, b: Tensor) -> Tensor:
    return logaddexp2_from_tensors(a, b)


def lerp(start: Tensor, end: Tensor, weight: Tensor | float) -> Tensor:
    return lerp_from_tensors(start, end, weight)


def addcmul(input: Tensor, t1: Tensor, t2: Tensor, value: float = 1.0) -> Tensor:
    return addcmul_from_tensors(input, t1, t2, value)


def addcdiv(input: Tensor, t1: Tensor, t2: Tensor, value: float = 1.0) -> Tensor:
    return addcdiv_from_tensors(input, t1, t2, value)


def xlogy(x: Tensor, y: Tensor) -> Tensor:
    return xlogy_from_tensors(x, y)


def copysign(a: Tensor, b: Tensor) -> Tensor:
    return copysign_from_tensors(a, b)


def nextafter(a: Tensor, b: Tensor) -> Tensor:
    return nextafter_from_tensors(a, b)


def floor_divide(a: Tensor, b: Tensor) -> Tensor:
    return floor_divide_from_tensors(a, b)


def true_divide(a: Tensor, b: Tensor) -> Tensor:
    return true_divide_from_tensors(a, b)


def logical_and(a: Tensor, b: Tensor) -> Tensor:
    return logical_and_from_tensors(a, b)


def logical_or(a: Tensor, b: Tensor) -> Tensor:
    return logical_or_from_tensors(a, b)


def logical_xor(a: Tensor, b: Tensor) -> Tensor:
    return logical_xor_from_tensors(a, b)


def bitwise_and(a: Tensor, b: Tensor) -> Tensor:
    return bitwise_and_from_tensors(a, b)


def bitwise_or(a: Tensor, b: Tensor) -> Tensor:
    return bitwise_or_from_tensors(a, b)


def bitwise_xor(a: Tensor, b: Tensor) -> Tensor:
    return bitwise_xor_from_tensors(a, b)


def bitwise_not(x: Tensor) -> Tensor:
    return x.bitwise_not()


def heaviside(input: Tensor, values: Tensor) -> Tensor:
    return heaviside_from_tensors(input, values)


def maximum(a: Tensor, b: Tensor) -> Tensor:
    return maximum_from_tensors(a, b)


def minimum(a: Tensor, b: Tensor) -> Tensor:
    return minimum_from_tensors(a, b)


def any(input: Tensor) -> Tensor:
    return any_from_tensor(input)


def all(input: Tensor) -> Tensor:
    return all_from_tensor(input)


def cumsum(input: Tensor) -> Tensor:
    return cumsum_from_tensor(input)


def cumprod(input: Tensor) -> Tensor:
    return cumprod_from_tensor(input)


def tril(input: Tensor, diagonal: int = 0) -> Tensor:
    return tril_from_tensor(input, diagonal)


def triu(input: Tensor, diagonal: int = 0) -> Tensor:
    return triu_from_tensor(input, diagonal)


def flip(input: Tensor, dims: Sequence[int]) -> Tensor:
    return flip_from_tensor(input, dims)


def eq(a: Tensor, b: Tensor) -> Tensor:
    return a.eq(b)


def ne(a: Tensor, b: Tensor) -> Tensor:
    return a.ne(b)


def lt(a: Tensor, b: Tensor) -> Tensor:
    return a.lt(b)


def le(a: Tensor, b: Tensor) -> Tensor:
    return a.le(b)


def gt(a: Tensor, b: Tensor) -> Tensor:
    return a.gt(b)


def ge(a: Tensor, b: Tensor) -> Tensor:
    return a.ge(b)


def sum(input: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    return input.sum(dim=dim, keepdim=keepdim)


def mean(input: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    return input.mean(dim=dim, keepdim=keepdim)


def prod(x: Tensor) -> Tensor:
    return x.prod()


def min(x: Tensor) -> Tensor:
    return x.min()


def max(x: Tensor) -> Tensor:
    return x.max()


def masked_select(input: Tensor, mask: Tensor) -> Tensor:
    return masked_select_from_tensor(input, mask=mask)


def masked_fill(input: Tensor, mask: Tensor, value: float) -> Tensor:
    return masked_fill_from_tensor(input, mask=mask, value=value)


def nonzero(input: Tensor) -> Tensor:
    return nonzero_from_tensor(input)


def roll(input: Tensor, shifts: int | list[int], dims: int | list[int] | None = None) -> Tensor:
    return roll_from_tensor(input, shifts, dims)


def equal(input: Tensor, other: Tensor) -> Tensor:
    return equal_from_tensors(input, other)


def allclose(input: Tensor, other: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> Tensor:
    return allclose_from_tensors(input, other, rtol, atol, equal_nan)


def isclose(input: Tensor, other: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> Tensor:
    return isclose_from_tensors(input, other, rtol, atol, equal_nan)


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    return softmax_from_tensor(input, dim)


def log_softmax(input: Tensor, dim: int = -1) -> Tensor:
    return log_softmax_from_tensor(input, dim)


def cholesky(x: Tensor) -> Tensor:
    return x.cholesky()


def lu(x: Tensor) -> tuple[Tensor, Tensor]:
    return x.lu()


def inv(x: Tensor) -> Tensor:
    return x.inv()


def det(x: Tensor) -> Tensor:
    return x.det()


def triangular_solve(a: Tensor, b: Tensor, upper: bool = False) -> Tensor:
    return a.triangular_solve(b, upper=upper)


# ── Save / Load ──────────────────────────────────────────────────

def save(obj: object, f: str) -> None:
    from ._save import save as _save
    _save(obj, f)


def load(f: str, map_location: object = None, weights_only: bool = False) -> object:
    from ._save import load as _load
    return _load(f, map_location=map_location, weights_only=weights_only)


# ── diag ─────────────────────────────────────────────────────────

def diag(x: Tensor) -> Tensor:
    return x.diag()


# Make submodules accessible
from torch import nn as nn
from torch import optim as optim
from torch import jit as jit
from torch import utils as utils
from torch import distributions as distributions
from torch import linalg as linalg

# DataLoader utilities
from torch.utils.data import (
    Dataset,
    TensorDataset,
    DataLoader,
    Sampler,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
    ConcatDataset,
    Subset,
    default_collate,
    default_convert,
)


# ── Shape ops ────────────────────────────────────────────────────

def split(tensor: Tensor, split_size: int | list[int], dim: int = 0) -> list[Tensor]:
    return tensor.split(split_size, dim=dim)


def chunk(tensor: Tensor, chunks: int, dim: int = 0) -> list[Tensor]:
    return tensor.chunk(chunks, dim=dim)


def repeat(tensor: Tensor, *sizes: int) -> Tensor:
    return tensor.repeat(*sizes)


def tile(tensor: Tensor, *sizes: int) -> Tensor:
    return tensor.repeat(*sizes)


def topk(tensor: Tensor, k: int, dim: int = -1, largest: bool = True) -> tuple[Tensor, Tensor]:
    return topk_from_tensor(tensor, k, dim=dim, largest=largest)


def sort(tensor: Tensor, dim: int = -1, descending: bool = False) -> tuple[Tensor, Tensor]:
    return sort_from_tensor(tensor, dim=dim, descending=descending)


def gather(tensor: Tensor, dim: int, index: Tensor) -> Tensor:
    return gather_from_tensor(tensor, dim, index)


def scatter(tensor: Tensor, dim: int, index: Tensor, src: Tensor | float) -> Tensor:
    return scatter_from_tensor(tensor, dim, index, src)


# ── Context managers (imported at top of file) ──────────────────
# no_grad, inference_mode, set_grad_enabled, is_grad_enabled, grad


# ── Dtype constants ──────────────────────────────────────────────

float32 = "float32"
float16 = "float16"
bfloat16 = "bfloat16"
float64 = "float64"
int8 = "int8"
int16 = "int16"
int32 = "int32"
int64 = "int64"
uint8 = "uint8"
uint16 = "uint16"
uint32 = "uint32"
uint64 = "uint64"
bool_ = "bool"

# PyTorch-style dtype aliases for compatibility.
long = int64
bool = bool_
double = float64
half = float16
int = int32
float = float32
byte = uint8

# Integer-width aliases aligned with PyTorch naming.
short = int16
char = int8


# Make submodules accessible
from torch import nn as nn
from torch import optim as optim
from torch import jit as jit
from torch import utils as utils
