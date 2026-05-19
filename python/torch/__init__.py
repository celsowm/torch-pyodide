from __future__ import annotations

from typing import Sequence

# Import autograd FIRST - must be before any other imports that might trigger Torch runtime
from .autograd import (
    no_grad,
    inference_mode,
    set_grad_enabled,
    is_grad_enabled,
    grad,
)

from . import cuda
from ._tensor import (
    Tensor,
    arange_from_values,
    cat_from_tensors,
    expand_from_tensor,
    full_from_shape,
    full_like_from_tensor,
    zeros_like_from_tensor,
    ones_like_from_tensor,
    empty_like_from_tensor,
    index_select_from_tensor,
    ones_from_shape,
    rand_from_shape,
    randn_from_shape,
    stack_from_tensors,
    tensor_from_data,
    where_from_tensors,
    zeros_from_shape,
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
    # New in this session
    empty_from_shape,
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
    "linspace",
    "logspace",
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "matmul",
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
]


def tensor(data: object, dtype: str = "float32", requires_grad: bool = False) -> Tensor:
    return tensor_from_data(data, dtype=dtype, requires_grad=requires_grad)


def zeros(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    return zeros_from_shape(shape, dtype=dtype)


def ones(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    return ones_from_shape(shape, dtype=dtype)

def rand(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    return rand_from_shape(shape, dtype=dtype)


def randn(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    return randn_from_shape(shape, dtype=dtype)


def arange(
    start: float,
    end: float | None = None,
    step: float = 1.0,
    dtype: str = "float32",
) -> Tensor:
    return arange_from_values(start=start, end=end, step=step, dtype=dtype)


def full(shape: int | Sequence[int], fill_value: float, dtype: str = "float32") -> Tensor:
    return full_from_shape(shape=shape, fill_value=fill_value, dtype=dtype)


def full_like(input: Tensor, fill_value: float, dtype: str | None = None) -> Tensor:
    return full_like_from_tensor(input, fill_value=fill_value, dtype=dtype)


def zeros_like(input: Tensor, dtype: str | None = None) -> Tensor:
    return zeros_like_from_tensor(input, dtype=dtype)


def ones_like(input: Tensor, dtype: str | None = None) -> Tensor:
    return ones_like_from_tensor(input, dtype=dtype)


def empty(shape: int | Sequence[int], dtype: str = "float32") -> Tensor:
    return empty_from_shape(shape, dtype=dtype)


def empty_like(input: Tensor, dtype: str | None = None) -> Tensor:
    return empty_like_from_tensor(input, dtype=dtype)


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


def clamp(x: Tensor, min: float, max: float) -> Tensor:
    return x.clamp(min=min, max=max)


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return where_from_tensors(condition, x, y)


def argmax(x: Tensor) -> Tensor:
    return x.argmax()


def argmin(x: Tensor) -> Tensor:
    return x.argmin()


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


# ── Einsum ────────────────────────────────────────────────────────

def einsum(equation: str, *operands: Tensor) -> Tensor:
    """Einstein summation convention. Supports 1, 2, or 3+ operands."""
    parts = equation.replace(" ", "").split("->")
    input_eq = parts[0]
    output_eq = parts[1] if len(parts) > 1 else ""
    input_terms = input_eq.split(",")
    ops = list(operands)

    if len(input_terms) == 1:
        return _einsum_single_op(input_terms[0], output_eq, ops[0])
    elif len(input_terms) == 2:
        return _einsum_two(input_terms[0], input_terms[1], output_eq, ops[0], ops[1])
    else:
        return _einsum_multi(input_terms, output_eq, ops)


def _einsum_two(a_idx: str, b_idx: str, output_eq: str, a: Tensor, b: Tensor) -> Tensor:
    sum_dims = set(a_idx) & set(b_idx) - set(output_eq)
    if len(a_idx) == 2 and len(b_idx) == 2 and len(sum_dims) == 1:
        sum_char = next(iter(sum_dims))
        a_sum_pos = a_idx.index(sum_char)
        b_sum_pos = b_idx.index(sum_char)
        if a_sum_pos == 0:
            a = a.transpose(0, 1)
        if b_sum_pos == 1:
            b = b.transpose(0, 1)
        return a.matmul(b)
    all_dims_str = "".join(dict.fromkeys(a_idx + b_idx))
    a_shape_map = {c: a.shape[a_idx.index(c)] for c in a_idx}
    b_shape_map = {c: b.shape[b_idx.index(c)] for c in b_idx}
    a_exp = a.reshape([a_shape_map.get(c, 1) if c in a_idx else 1 for c in all_dims_str])
    b_exp = b.reshape([b_shape_map.get(c, 1) if c in b_idx else 1 for c in all_dims_str])
    expanded = a_exp * b_exp
    sum_dims_list = [all_dims_str.index(c) for c in sum_dims]
    if sum_dims_list:
        result = expanded
        for d in reversed(sorted(sum_dims_list)):
            result = result.sum(dim=d)
        if output_eq:
            perm = [all_dims_str.index(c) for c in output_eq if c in all_dims_str]
            if perm:
                result = result.permute(perm)
    else:
        result = expanded
    return result


def _einsum_multi(input_terms: list[str], output_eq: str, ops: list[Tensor]) -> Tensor:
    """3+ operands: contract pairs iteratively, then final reduce/permute."""
    current_ops = [o for o in ops]
    current_indices = [s for s in input_terms]

    while len(current_ops) > 2:
        # Contract first two ops
        a_idx = current_indices[0]
        b_idx = current_indices[1]
        a_op = current_ops[0]
        b_op = current_ops[1]
        # Determine what dims to keep for intermediate
        all_chars = "".join(dict.fromkeys(a_idx + b_idx))
        common = set(a_idx) & set(b_idx)
        remaining = set()
        for idx in current_indices[2:]:
            remaining.update(idx)
        remaining.update(output_eq)
        intermediate_chars = "".join(c for c in all_chars if c not in (common - remaining))
        result = _einsum_two(a_idx, b_idx, intermediate_chars, a_op, b_op)
        current_ops = [result] + current_ops[2:]
        current_indices = [intermediate_chars] + current_indices[2:]

    if len(current_ops) == 2:
        result = _einsum_two(current_indices[0], current_indices[1], output_eq, current_ops[0], current_ops[1])
    else:
        result = current_ops[0]
        final_idx = current_indices[0]
        sum_chars = set(final_idx) - set(output_eq)
        for c in sum_chars:
            dim = final_idx.index(c)
            result = result.sum(dim=dim)
        final_idx = "".join(c for c in final_idx if c not in sum_chars)
        if final_idx != output_eq and len(final_idx) == len(output_eq):
            perm = [final_idx.index(c) for c in output_eq]
            result = result.permute(perm)
    return result


def _einsum_single_op(input_str: str, output_str: str, x: Tensor) -> Tensor:
    """Handle single operand einsum like 'ii->i' (diagonal) or 'ij->ji' (transpose)."""
    if len(input_str) == 2 and len(output_str) == 2:
        if input_str == output_str:
            return x
        # Transpose
        perm = [input_str.index(c) for c in output_str]
        return x.permute(perm)
    if len(input_str) == 2 and len(output_str) == 1 and input_str[0] == input_str[1]:
        # ii -> i: diagonal
        n = x.shape[0]
        vals = [x.tolist()[i * n + i] for i in range(n)]
        from ._tensor import tensor_from_data
        return tensor_from_data(vals, x.dtype)
    if len(input_str) == 2 and len(output_str) == 0:
        # ij -> : trace (sum of diagonal)
        return x.sum()
    return x


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


# ── Creation ops ─────────────────────────────────────────────────

def eye(n: int, m: int | None = None, dtype: str = "float32") -> Tensor:
    rows = n
    cols = m if m is not None else n
    data: list[float] = []
    for i in range(rows):
        for j in range(cols):
            data.append(1.0 if i == j else 0.0)
    return tensor_from_data(data, [rows, cols], dtype=dtype)


def randint(low: int, high: int | None = None, size: int | Sequence[int] | None = None, dtype: str = "int64") -> Tensor:
    if high is None:
        low, high = 0, low
    if size is None:
        size = [1]
    if isinstance(size, int):
        size = [size]
    r = rand(list(size))
    span = float(high - low)
    scaled = r.mul(span).add(float(low))
    return scaled.to(dtype)


def randperm(n: int, dtype: str = "int64") -> Tensor:
    import random
    indices = list(range(n))
    random.shuffle(indices)
    return tensor_from_data(indices, [n], dtype=dtype)


def linspace(start: float, end: float, steps: int, dtype: str = "float32") -> Tensor:
    if steps < 2:
        return full([steps], start, dtype=dtype)
    step = (end - start) / (steps - 1)
    return arange(start=start, end=end + step * 0.5, step=step, dtype=dtype)


def logspace(start: float, end: float, steps: int, dtype: str = "float32") -> Tensor:
    return linspace(start, end, steps, dtype=dtype).pow(10.0)


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
int32 = "int32"
int64 = "int64"
uint8 = "uint8"
bool_ = "bool"


# Make submodules accessible
from torch import nn as nn
from torch import optim as optim
from torch import jit as jit
from torch import utils as utils
