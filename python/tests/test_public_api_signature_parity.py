from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import torch


ROOT = Path(__file__).resolve().parents[1]

IGNORED_UPSTREAM_PARAMS = {
    "_use_new_zipfile_serialization",
    "alpha",
    "base",
    "decimals",
    "device",
    "generator",
    "layout",
    "memory_format",
    "mmap",
    "out",
    "out_dtype",
    "pickle_load_args",
    "pickle_module",
    "pickle_protocol",
    "pin_memory",
    "rounding_mode",
    "sparse_grad",
    "stable",
}

KNOWN_SIGNATURE_GAPS = {
    "abs",
    "acos",
    "acosh",
    "add",
    "addcdiv",
    "addcmul",
    "arange",
    "argmax",
    "argmin",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bernoulli",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "bmm",
    "ceil",
    "chunk",
    "clamp",
    "copysign",
    "cos",
    "cosh",
    "cumprod",
    "cumsum",
    "deg2rad",
    "diag",
    "digamma",
    "div",
    "dot",
    "empty_like",
    "eq",
    "erf",
    "erfc",
    "exp",
    "exp2",
    "exponential",
    "expm1",
    "eye",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "frac",
    "full_like",
    "gather",
    "ge",
    "gt",
    "histogram",
    "hypot",
    "i0",
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "kthvalue",
    "le",
    "lerp",
    "lgamma",
    "linspace",
    "log",
    "log10",
    "log1p",
    "log2",
    "log_normal",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logspace",
    "lt",
    "matmul",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "mode",
    "mm",
    "mul",
    "mv",
    "ne",
    "neg",
    "nextafter",
    "normal",
    "nonzero",
    "ones_like",
    "outer",
    "pow",
    "prod",
    "quantile",
    "rad2deg",
    "randint",
    "randperm",
    "reciprocal",
    "remainder",
    "round",
    "rsqrt",
    "scatter",
    "searchsorted",
    "sgn",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "softmax",
    "sort",
    "sqrt",
    "square",
    "sub",
    "sum",
    "tan",
    "tanh",
    "tile",
    "topk",
    "true_divide",
    "trunc",
    "unique",
    "where",
    "xlogy",
    "zeros_like",
}


def _split_signature_params(signature_line: str) -> list[str]:
    if not signature_line:
        return []

    params = signature_line[signature_line.find("(") + 1 :]
    end = params.find(") ->")
    params = params[:end] if end >= 0 else params.rstrip(")")

    parts: list[str] = []
    current = ""
    depth = 0
    for char in params:
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
            continue
        current += char
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
    if current.strip():
        parts.append(current.strip())
    return parts


def _param_name(raw_param: str) -> str:
    name = raw_param.strip().replace("\\*", "*")
    name = name.split("=", 1)[0].strip()
    name = name.split(":", 1)[0].strip()
    return name.lstrip("*")


def _installed_torch_signature_docs(names: list[str]) -> dict[str, str]:
    script = r"""
import json
import torch

names = json.loads(input())
signatures = {}
for name in names:
    obj = getattr(torch, name, None)
    doc = getattr(obj, "__doc__", "") or ""
    signature = ""
    for line in doc.splitlines():
        candidate = line.strip().replace("\\*", "*")
        if candidate.startswith(f"{name}(") or candidate.startswith(f"torch.{name}("):
            signature = candidate
            break
    signatures[name] = signature
print(json.dumps({"torch_file": torch.__file__, "signatures": signatures}))
"""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tempfile.gettempdir(),
        env=env,
        input=json.dumps(names),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        pytest.skip(f"installed PyTorch is not importable: {process.stderr.strip()}")

    payload = json.loads(process.stdout)
    torch_file = Path(payload["torch_file"]).resolve()
    if ROOT in torch_file.parents:
        pytest.skip(f"installed PyTorch resolved to local source tree: {torch_file}")
    return payload["signatures"]


def _local_public_callables() -> list[str]:
    return [
        name
        for name in getattr(torch, "__all__", [])
        if not name.startswith("_") and name != "Tensor" and callable(getattr(torch, name, None))
    ]


def _signature_gaps_against_installed_torch() -> dict[str, list[str]]:
    names = _local_public_callables()
    upstream_signatures = _installed_torch_signature_docs(names)
    gaps: dict[str, list[str]] = {}

    for name in names:
        upstream_params = _split_signature_params(upstream_signatures.get(name, ""))
        if not upstream_params:
            continue

        local_signature = inspect.signature(getattr(torch, name))
        local_params = local_signature.parameters
        issues: list[str] = []

        for raw_param in upstream_params:
            if raw_param == "*":
                continue

            upstream_name = _param_name(raw_param)
            if upstream_name in IGNORED_UPSTREAM_PARAMS:
                continue

            if raw_param.startswith("**"):
                has_param = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD and p.name == upstream_name
                    for p in local_params.values()
                )
            elif raw_param.startswith("*"):
                has_param = any(
                    p.kind == inspect.Parameter.VAR_POSITIONAL and p.name == upstream_name
                    for p in local_params.values()
                )
            else:
                has_param = upstream_name in local_params

            if not has_param:
                issues.append(f"missing `{raw_param}` from upstream `{upstream_signatures[name]}`")

        if issues:
            gaps[name] = issues

    return gaps


def test_public_api_signatures_match_installed_torch_without_new_gaps():
    gaps = _signature_gaps_against_installed_torch()
    unexpected = sorted(set(gaps) - KNOWN_SIGNATURE_GAPS)

    assert not unexpected, json.dumps(
        {name: gaps[name] for name in unexpected},
        indent=2,
        sort_keys=True,
    )
