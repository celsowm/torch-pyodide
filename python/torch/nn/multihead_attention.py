from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn.modules import Module, Linear, Dropout


class MultiheadAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = True, batch_first: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.dropout = dropout
        self.batch_first = batch_first

        self.in_proj_weight = torch.empty((3 * embed_dim, embed_dim))
        self.in_proj_bias = torch.zeros((3 * embed_dim,))
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                key_padding_mask: Tensor | None = None,
                attn_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        tgt_len, batch_size, embed_dim = query.shape
        src_len = key.shape[0]

        # Linear projection
        qkv = query.matmul(self.in_proj_weight.T) + self.in_proj_bias
        q, k_, v = qkv.chunk(3, dim=-1)

        if key is not query or value is not query:
            k_ = key.matmul(self.in_proj_weight.T[:, embed_dim:2*embed_dim]) + self.in_proj_bias[embed_dim:2*embed_dim]
            v = value.matmul(self.in_proj_weight.T[:, 2*embed_dim:]) + self.in_proj_bias[2*embed_dim:]

        # Reshape for multi-head
        q = q.reshape(tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k_ = k_.reshape(src_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(src_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = q.matmul(k_.transpose(1, 2)) * scale

        if attn_mask is not None:
            attn = attn + attn_mask

        if key_padding_mask is not None:
            mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, tgt_len, -1)
            mask_expanded = mask_expanded.reshape(batch_size * self.num_heads, tgt_len, src_len)
            attn = attn.masked_fill(mask_expanded, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        if self.dropout > 0:
            attn = torch.nn.functional.dropout(attn, self.dropout, self.training)

        out = attn.matmul(v)
        out = out.transpose(0, 1).reshape(tgt_len, batch_size, embed_dim)
        if self.batch_first:
            out = out.transpose(0, 1)

        out = self.out_proj(out)
        return out, attn
