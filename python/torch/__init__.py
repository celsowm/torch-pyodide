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
    index_add_from_tensor,
    index_copy_from_tensor,
    index_fill_from_tensor,
    take_from_tensor,
    unfold_from_tensor,
    cdist_from_tensor,
    pdist_from_tensor,
    scatter_add_safe_from_tensor,
    searchsorted_from_tensor,
    kthvalue_from_tensor,
    median_from_tensor,
    quantile_from_tensor,
    mode_from_tensor,
    unique_from_tensor,
    histogram_from_tensor,
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
    "index_add",
    "index_copy",
    "index_fill",
    "take",
    "unfold",
    "cdist",
    "pdist",
    "searchsorted",
    "kthvalue",
    "median",
    "quantile",
    "mode",
    "unique",
    "histogram",
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
    "amax",
    "amin",
    "aminmax",
    "logsumexp",
    "var",
    "std",
    "var_mean",
    "std_mean",
    "nan_to_num",
    "count_nonzero",
    "unbind",
    "movedim",
    "moveaxis",
    "swapaxes",
    "swapdims",
    "ravel",
    "broadcast_to",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "hstack",
    "vstack",
    "row_stack",
    "dstack",
    "column_stack",
    "flipud",
    "fliplr",
    "diff",
    "trace",
    "diagflat",
    "dist",
    "float_power",
    "clamp_min",
    "clamp_max",
    "clip",
    "take_along_dim",
    "inner",
    "vdot",
    "kron",
    "tensordot",
    "signbit",
    "positive",
    "negative",
    "nansum",
    "nanmean",
    "isreal",
    "narrow",
    "cross",
    "rot90",
    "renorm",
    "broadcast_tensors",
    "tensor_split",
    "hsplit",
    "vsplit",
    "dsplit",
    "addmm",
    "addmv",
    "baddbmm",
    "addbmm",
    "chain_matmul",
    "meshgrid",
    "cartesian_prod",
    "diag_embed",
    "block_diag",
    "tril_indices",
    "triu_indices",
    "trapezoid",
    "trapz",
    "sinc",
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


def complex(real: Tensor, imag: Tensor) -> Tensor:
    from ._complex import complex as _complex
    return _complex(real, imag)


def polar(abs: Tensor, angle: Tensor) -> Tensor:
    from ._complex import polar as _polar
    return _polar(abs, angle)


def view_as_complex(x: Tensor) -> Tensor:
    from ._complex import view_as_complex as _vac
    return _vac(x)


def view_as_real(x: Tensor) -> Tensor:
    from ._complex import view_as_real as _var
    return _var(x)


def real(x: Tensor) -> Tensor:
    from ._complex import real as _real
    return _real(x)


def imag(x: Tensor) -> Tensor:
    from ._complex import imag as _imag
    return _imag(x)


def conj(x: Tensor) -> Tensor:
    from ._complex import conj as _conj
    return _conj(x)


def angle(x: Tensor) -> Tensor:
    from ._complex import angle as _angle
    return _angle(x)


def is_complex(x: Tensor) -> bool:
    from ._complex import is_complex as _is_complex
    return _is_complex(x)


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


def index_add(input: Tensor, dim: int, index: Tensor, source: Tensor) -> Tensor:
    return index_add_from_tensor(input, dim=dim, index=index, source=source)


def index_copy(input: Tensor, dim: int, index: Tensor, source: Tensor) -> Tensor:
    return index_copy_from_tensor(input, dim=dim, index=index, source=source)


def index_fill(input: Tensor, dim: int, index: Tensor, value: Tensor | float) -> Tensor:
    return index_fill_from_tensor(input, dim=dim, index=index, value=value)


def take(input: Tensor, index: Tensor) -> Tensor:
    return take_from_tensor(input, index=index)


def unfold(input: Tensor, dimension: int, size: int, step: int = 1) -> Tensor:
    return unfold_from_tensor(input, dimension=dimension, size=size, step=step)


def cdist(x1: Tensor, x2: Tensor, p: float = 2.0) -> Tensor:
    return cdist_from_tensor(x1, x2, p=p)


def pdist(input: Tensor, p: float = 2.0) -> Tensor:
    return pdist_from_tensor(input, p=p)


def searchsorted(sorted_sequence: Tensor, values: Tensor, right: bool = False) -> Tensor:
    return searchsorted_from_tensor(sorted_sequence, values, right=right)


def kthvalue(input: Tensor, k: int, dim: int = -1):
    return kthvalue_from_tensor(input, k, dim=dim)


def median(input: Tensor, dim=None):
    return median_from_tensor(input, dim)


def quantile(input, q, dim: int = -1):
    return quantile_from_tensor(input, q, dim=dim)


def mode(input: Tensor, dim=None):
    return mode_from_tensor(input, dim)


def unique(input: Tensor, return_counts: bool = False, sorted: bool = True, dim=None):
    return unique_from_tensor(input, return_counts=return_counts, sorted=sorted, dim=dim)


def histogram(input: Tensor, bins: int, range=None):
    return histogram_from_tensor(input, bins, range=range)


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


def min(x: Tensor, dim: int | None = None, keepdim: bool = False):
    if dim is not None:
        return x.min(dim=dim, keepdim=keepdim)
    return x.min()


def max(x: Tensor, dim: int | None = None, keepdim: bool = False):
    if dim is not None:
        return x.max(dim=dim, keepdim=keepdim)
    return x.max()


def amax(input: Tensor, dim=None, keepdim: bool = False) -> Tensor:
    return input.amax(dim, keepdim)


def amin(input: Tensor, dim=None, keepdim: bool = False) -> Tensor:
    return input.amin(dim, keepdim)


def aminmax(input: Tensor, dim=None, keepdim: bool = False):
    return (input.amin(dim, keepdim), input.amax(dim, keepdim))


def logsumexp(input: Tensor, dim, keepdim: bool = False) -> Tensor:
    return input.logsumexp(dim, keepdim)


def var(input: Tensor, dim=None, keepdim: bool = False, correction: int = 1, unbiased: bool | None = None) -> Tensor:
    return input.var(dim=dim, keepdim=keepdim, correction=correction, unbiased=unbiased)


def std(input: Tensor, dim=None, keepdim: bool = False, correction: int = 1, unbiased: bool | None = None) -> Tensor:
    return input.std(dim=dim, keepdim=keepdim, correction=correction, unbiased=unbiased)


def var_mean(input: Tensor, dim=None, keepdim: bool = False, correction: int = 1, unbiased: bool | None = None):
    return (input.var(dim=dim, keepdim=keepdim, correction=correction, unbiased=unbiased), input.mean(dim=dim, keepdim=keepdim) if dim is not None else input.mean())


def std_mean(input: Tensor, dim=None, keepdim: bool = False, correction: int = 1, unbiased: bool | None = None):
    return (input.std(dim=dim, keepdim=keepdim, correction=correction, unbiased=unbiased), input.mean(dim=dim, keepdim=keepdim) if dim is not None else input.mean())


def nan_to_num(input: Tensor, nan: float = 0.0, posinf=None, neginf=None) -> Tensor:
    return input.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)


def count_nonzero(input: Tensor, dim=None) -> Tensor:
    return input.count_nonzero(dim)


def unbind(input: Tensor, dim: int = 0):
    return input.unbind(dim)


def movedim(input: Tensor, source, destination) -> Tensor:
    return input.movedim(source, destination)


def moveaxis(input: Tensor, source, destination) -> Tensor:
    return input.movedim(source, destination)


def swapaxes(input: Tensor, axis0: int, axis1: int) -> Tensor:
    return input.transpose(axis0, axis1)


def swapdims(input: Tensor, dim0: int, dim1: int) -> Tensor:
    return input.transpose(dim0, dim1)


def ravel(input: Tensor) -> Tensor:
    return input.reshape([-1])


def broadcast_to(input: Tensor, shape) -> Tensor:
    shape = list(shape)
    while len(input._shape) < len(shape):
        input = input.unsqueeze(0)
    return input.expand(*shape)


def atleast_1d(*tensors):
    def fix(t):
        return t.reshape([1]) if len(t._shape) == 0 else t
    res = [fix(t) for t in tensors]
    return res[0] if len(res) == 1 else tuple(res)


def atleast_2d(*tensors):
    def fix(t):
        nd = len(t._shape)
        if nd == 0:
            return t.reshape([1, 1])
        if nd == 1:
            return t.reshape([1, t._shape[0]])
        return t
    res = [fix(t) for t in tensors]
    return res[0] if len(res) == 1 else tuple(res)


def atleast_3d(*tensors):
    def fix(t):
        nd = len(t._shape)
        if nd == 0:
            return t.reshape([1, 1, 1])
        if nd == 1:
            return t.reshape([1, t._shape[0], 1])
        if nd == 2:
            return t.reshape([t._shape[0], t._shape[1], 1])
        return t
    res = [fix(t) for t in tensors]
    return res[0] if len(res) == 1 else tuple(res)


def hstack(tensors) -> Tensor:
    tensors = list(tensors)
    if all(len(t._shape) == 1 for t in tensors):
        return cat(tensors, dim=0)
    return cat(tensors, dim=1)


def vstack(tensors) -> Tensor:
    return cat([atleast_2d(t) for t in tensors], dim=0)


def row_stack(tensors) -> Tensor:
    return vstack(tensors)


def dstack(tensors) -> Tensor:
    return cat([atleast_3d(t) for t in tensors], dim=2)


def column_stack(tensors) -> Tensor:
    cols = []
    for t in tensors:
        if len(t._shape) == 1:
            cols.append(t.reshape([t._shape[0], 1]))
        else:
            cols.append(t)
    return cat(cols, dim=1)


def flipud(input: Tensor) -> Tensor:
    return input.flip([0])


def fliplr(input: Tensor) -> Tensor:
    return input.flip([1])


def diff(input: Tensor, n: int = 1, dim: int = -1) -> Tensor:
    d = dim if dim >= 0 else dim + len(input._shape)
    out = input
    for _ in range(n):
        size = out._shape[d]
        a = out.narrow(d, 1, size - 1)
        b = out.narrow(d, 0, size - 1)
        out = a.sub(b)
    return out


def trace(input: Tensor) -> Tensor:
    return input.diag().sum()


def diagflat(input: Tensor, offset: int = 0) -> Tensor:
    return input.reshape([-1]).diag()


def dist(input: Tensor, other: Tensor, p: float = 2) -> Tensor:
    return norm(input.sub(other), p=p)


def float_power(input: Tensor, exponent) -> Tensor:
    return input.pow(exponent)


def clamp_min(input: Tensor, min) -> Tensor:
    return input.clamp(min=min)


def clamp_max(input: Tensor, max) -> Tensor:
    return input.clamp(max=max)


def clip(input: Tensor, min=None, max=None) -> Tensor:
    return input.clamp(min=min, max=max)


def take_along_dim(input: Tensor, indices: Tensor, dim: int) -> Tensor:
    return input.gather(dim, indices)


def inner(input: Tensor, other: Tensor) -> Tensor:
    if len(input._shape) == 1 and len(other._shape) == 1:
        return input.mul(other).sum()
    return tensordot(input, other, dims=([len(input._shape) - 1], [len(other._shape) - 1]))


def vdot(input: Tensor, other: Tensor) -> Tensor:
    return input.reshape([-1]).mul(other.reshape([-1])).sum()


def kron(input: Tensor, other: Tensor) -> Tensor:
    a = input
    b = other
    while len(a._shape) < len(b._shape):
        a = a.unsqueeze(0)
    while len(b._shape) < len(a._shape):
        b = b.unsqueeze(0)
    nd = len(a._shape)
    a_exp_shape = []
    b_exp_shape = []
    out_shape = []
    for i in range(nd):
        a_exp_shape += [a._shape[i], 1]
        b_exp_shape += [1, b._shape[i]]
        out_shape.append(a._shape[i] * b._shape[i])
    a2 = a.reshape(a_exp_shape)
    b2 = b.reshape(b_exp_shape)
    return a2.mul(b2).reshape(out_shape)


def tensordot(a: Tensor, b: Tensor, dims=2) -> Tensor:
    a_shape = list(a._shape)
    b_shape = list(b._shape)
    na = len(a_shape)
    nb = len(b_shape)
    if isinstance(dims, int):
        a_axes = list(range(na - dims, na))
        b_axes = list(range(dims))
    else:
        a_axes = [d if d >= 0 else d + na for d in dims[0]]
        b_axes = [d if d >= 0 else d + nb for d in dims[1]]
    a_free = [i for i in range(na) if i not in a_axes]
    b_free = [i for i in range(nb) if i not in b_axes]
    a_perm = a.permute(a_free + a_axes)
    b_perm = b.permute(b_axes + b_free)
    a_free_size = 1
    for i in a_free:
        a_free_size *= a_shape[i]
    b_free_size = 1
    for i in b_free:
        b_free_size *= b_shape[i]
    k = 1
    for i in a_axes:
        k *= a_shape[i]
    a2 = a_perm.reshape([a_free_size, k])
    b2 = b_perm.reshape([k, b_free_size])
    out = a2.matmul(b2)
    out_shape = [a_shape[i] for i in a_free] + [b_shape[i] for i in b_free]
    return out.reshape(out_shape) if out_shape else out.reshape([])


def signbit(input: Tensor) -> Tensor:
    return input.signbit()


def positive(input: Tensor) -> Tensor:
    return input


def negative(input: Tensor) -> Tensor:
    return input.neg()


def nansum(input: Tensor, dim=None, keepdim: bool = False) -> Tensor:
    cleaned = input.nan_to_num(nan=0.0, posinf=None, neginf=None)
    return cleaned.sum(dim=dim, keepdim=keepdim) if dim is not None else cleaned.sum()


def nanmean(input: Tensor, dim=None, keepdim: bool = False) -> Tensor:
    mask = input.isnan()
    cleaned = input.nan_to_num(nan=0.0)
    ones = mask.logical_not().to(input._dtype)
    if dim is None:
        return cleaned.sum().div(ones.sum())
    return cleaned.sum(dim=dim, keepdim=keepdim).div(ones.sum(dim=dim, keepdim=keepdim))


def isreal(input: Tensor) -> Tensor:
    if input.is_complex():
        return input.imag == 0
    return ones(list(input._shape), dtype="bool") if input._shape else ones([], dtype="bool")


def narrow(input: Tensor, dim: int, start: int, length: int) -> Tensor:
    return input.narrow(dim, start, length)


def cross(input: Tensor, other: Tensor, dim: int | None = None) -> Tensor:
    if dim is None:
        dim = next((i for i, s in enumerate(input._shape) if s == 3), -1)
    from .linalg import cross as _lcross
    return _lcross(input, other, dim=dim)


def rot90(input: Tensor, k: int = 1, dims=(0, 1)) -> Tensor:
    d0, d1 = dims[0], dims[1]
    out = input
    for _ in range(k % 4):
        out = out.flip([d1]).transpose(d0, d1)
    return out


def renorm(input: Tensor, p: float, dim: int, maxnorm: float) -> Tensor:
    nd = len(input._shape)
    d = dim if dim >= 0 else dim + nd
    reduce_dims = [i for i in range(nd) if i != d]
    norms = input.abs().pow(p)
    for rd in reduce_dims:
        norms = norms.sum(dim=rd, keepdim=True)
    norms = norms.pow(1.0 / p)
    factor = norms.add(1e-7).pow(-1.0).mul(float(maxnorm)).clamp(max=1.0)
    return input.mul(factor)


def broadcast_tensors(*tensors):
    ndim = _builtins.max(len(t._shape) for t in tensors)
    shapes = []
    for t in tensors:
        s = [1] * (ndim - len(t._shape)) + list(t._shape)
        shapes.append(s)
    target = []
    for i in range(ndim):
        target.append(_builtins.max(s[i] for s in shapes))
    return tuple(broadcast_to(t, target) for t in tensors)


def tensor_split(input: Tensor, indices_or_sections, dim: int = 0):
    d = dim if dim >= 0 else dim + len(input._shape)
    size = input._shape[d]
    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        base = size // n
        rem = size % n
        result = []
        start = 0
        for i in range(n):
            length = base + (1 if i < rem else 0)
            result.append(input.narrow(d, start, length))
            start += length
        return tuple(result)
    bounds = list(indices_or_sections)
    result = []
    prev = 0
    for b in bounds:
        b = _builtins.min(b, size)
        result.append(input.narrow(d, prev, _builtins.max(0, b - prev)))
        prev = b
    result.append(input.narrow(d, prev, size - prev))
    return tuple(result)


def hsplit(input: Tensor, indices_or_sections):
    dim = 0 if len(input._shape) == 1 else 1
    return tensor_split(input, indices_or_sections, dim=dim)


def vsplit(input: Tensor, indices_or_sections):
    return tensor_split(input, indices_or_sections, dim=0)


def dsplit(input: Tensor, indices_or_sections):
    return tensor_split(input, indices_or_sections, dim=2)


def addmm(input: Tensor, mat1: Tensor, mat2: Tensor, beta: float = 1, alpha: float = 1) -> Tensor:
    return input.mul(beta).add(mat1.matmul(mat2).mul(alpha))


def addmv(input: Tensor, mat: Tensor, vec: Tensor, beta: float = 1, alpha: float = 1) -> Tensor:
    return input.mul(beta).add(mv(mat, vec).mul(alpha))


def baddbmm(input: Tensor, batch1: Tensor, batch2: Tensor, beta: float = 1, alpha: float = 1) -> Tensor:
    return input.mul(beta).add(bmm(batch1, batch2).mul(alpha))


def addbmm(input: Tensor, batch1: Tensor, batch2: Tensor, beta: float = 1, alpha: float = 1) -> Tensor:
    reduced = bmm(batch1, batch2).sum(dim=0)
    return input.mul(beta).add(reduced.mul(alpha))


def chain_matmul(*matrices) -> Tensor:
    mats = list(matrices)
    out = mats[0]
    for m in mats[1:]:
        out = out.matmul(m)
    return out


def meshgrid(*tensors, indexing: str = "ij"):
    n = len(tensors)
    sizes = [t._shape[0] for t in tensors]
    outs = []
    for i, t in enumerate(tensors):
        shape = [1] * n
        shape[i] = sizes[i]
        outs.append(t.reshape(shape).expand(*sizes))
    if indexing == "xy" and n >= 2:
        outs = [o.transpose(0, 1) for o in outs]
    return tuple(outs)


def cartesian_prod(*tensors):
    if len(tensors) == 1:
        return tensors[0]
    grids = meshgrid(*tensors, indexing="ij")
    flat = [g.reshape([-1]) for g in grids]
    return stack(flat, dim=1)


def diag_embed(input: Tensor, offset: int = 0, dim1: int = -2, dim2: int = -1) -> Tensor:
    n = input._shape[-1]
    return input.unsqueeze(-1).mul(eye(n, dtype=input._dtype))


def block_diag(*tensors) -> Tensor:
    mats = [atleast_2d(t) for t in tensors]
    total_c = _builtins.sum(m._shape[1] for m in mats)
    rows = []
    coffset = 0
    for m in mats:
        r, c = m._shape[0], m._shape[1]
        parts = []
        if coffset > 0:
            parts.append(zeros([r, coffset], dtype=m._dtype))
        parts.append(m)
        right = total_c - coffset - c
        if right > 0:
            parts.append(zeros([r, right], dtype=m._dtype))
        rows.append(parts[0] if len(parts) == 1 else cat(parts, dim=1))
        coffset += c
    return rows[0] if len(rows) == 1 else cat(rows, dim=0)


def tril_indices(row: int, col: int, offset: int = 0, dtype: str = "int64", device=None) -> Tensor:
    rs = []
    cs = []
    for i in range(row):
        for j in range(col):
            if j - i <= offset:
                rs.append(i)
                cs.append(j)
    return tensor([rs, cs], dtype=dtype)


def triu_indices(row: int, col: int, offset: int = 0, dtype: str = "int64", device=None) -> Tensor:
    rs = []
    cs = []
    for i in range(row):
        for j in range(col):
            if j - i >= offset:
                rs.append(i)
                cs.append(j)
    return tensor([rs, cs], dtype=dtype)


def trapezoid(y: Tensor, x: Tensor | None = None, dx: float | None = None, dim: int = -1) -> Tensor:
    d = dim if dim >= 0 else dim + len(y._shape)
    n = y._shape[d]
    left = y.narrow(d, 0, n - 1)
    right = y.narrow(d, 1, n - 1)
    avg = left.add(right).mul(0.5)
    if x is not None:
        xd = x.narrow(0, 1, x._shape[0] - 1).sub(x.narrow(0, 0, x._shape[0] - 1))
        shape = [1] * len(y._shape)
        shape[d] = xd._shape[0]
        avg = avg.mul(xd.reshape(shape))
        return avg.sum(dim=d)
    spacing = 1.0 if dx is None else float(dx)
    return avg.sum(dim=d).mul(spacing)


def trapz(y: Tensor, x: Tensor | None = None, dx: float | None = None, dim: int = -1) -> Tensor:
    return trapezoid(y, x=x, dx=dx, dim=dim)


def sinc(input: Tensor) -> Tensor:
    from .special import sinc as _sinc
    return _sinc(input)


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
from torch import special as special
from torch import fft as fft

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
complex64 = "complex64"
complex128 = "complex128"
cfloat = complex64
cdouble = complex128

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
