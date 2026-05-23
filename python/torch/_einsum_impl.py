from __future__ import annotations

from ._tensor import Tensor
from .tensor_factories_ops import tensor_from_data


def einsum(equation: str, *operands: Tensor) -> Tensor:
    """Einstein summation convention. Supports 1, 2, or 3+ operands."""
    parts = equation.replace(" ", "").split("->")
    input_eq = parts[0]
    output_eq = parts[1] if len(parts) > 1 else ""
    input_terms = input_eq.split(",")
    ops = list(operands)

    if len(input_terms) == 1:
        return _einsum_single_op(input_terms[0], output_eq, ops[0])
    if len(input_terms) == 2:
        return _einsum_two(input_terms[0], input_terms[1], output_eq, ops[0], ops[1])
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
            remaining_chars = "".join(c for c in all_dims_str if c not in sum_dims)
            perm = [remaining_chars.index(c) for c in output_eq if c in remaining_chars]
            if perm:
                result = result.permute(perm)
    else:
        result = expanded
    return result


def _einsum_multi(input_terms: list[str], output_eq: str, ops: list[Tensor]) -> Tensor:
    current_ops = [o for o in ops]
    current_indices = [s for s in input_terms]

    while len(current_ops) > 2:
        a_idx = current_indices[0]
        b_idx = current_indices[1]
        a_op = current_ops[0]
        b_op = current_ops[1]
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
    if len(input_str) == 2 and len(output_str) == 2:
        if input_str == output_str:
            return x
        perm = [input_str.index(c) for c in output_str]
        return x.permute(perm)
    if len(input_str) == 2 and len(output_str) == 1 and input_str[0] == input_str[1]:
        n = x.shape[0]
        vals = [x.tolist()[i * n + i] for i in range(n)]
        return tensor_from_data(vals, x.dtype)
    if len(input_str) == 2 and len(output_str) == 0:
        return x.sum()
    return x
