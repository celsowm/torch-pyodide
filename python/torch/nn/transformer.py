from __future__ import annotations

import math
import copy as _copy

import torch
from torch import Tensor
from torch.nn.modules import Module, Linear, Dropout, LayerNorm
from torch.nn.multihead_attention import MultiheadAttention


class TransformerEncoderLayer(Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu",
                 batch_first: bool = True) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = activation

    def forward(self, src: Tensor, src_mask: Tensor | None = None,
                src_key_padding_mask: Tensor | None = None) -> Tensor:
        attn_out, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                     key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        if self.activation == "relu":
            ff_out = self.linear2(self.dropout(self.linear1(src).relu()))
        else:
            ff_out = self.linear2(self.dropout(self.linear1(src).gelu()))
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu",
                 batch_first: bool = True) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = activation

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Tensor | None = None, memory_mask: Tensor | None = None,
                tgt_key_padding_mask: Tensor | None = None,
                memory_key_padding_mask: Tensor | None = None) -> Tensor:
        attn_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                     key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(attn_out)
        tgt = self.norm1(tgt)
        attn_out2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                           key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(attn_out2)
        tgt = self.norm2(tgt)
        if self.activation == "relu":
            ff_out = self.linear2(self.dropout(self.linear1(tgt).relu()))
        else:
            ff_out = self.linear2(self.dropout(self.linear1(tgt).gelu()))
        tgt = tgt + self.dropout3(ff_out)
        tgt = self.norm3(tgt)
        return tgt


class TransformerEncoder(Module):
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = [_copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        self.num_layers = num_layers

    def forward(self, src: Tensor, mask: Tensor | None = None,
                src_key_padding_mask: Tensor | None = None) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output


class TransformerDecoder(Module):
    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = [_copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        self.num_layers = num_layers

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Tensor | None = None, memory_mask: Tensor | None = None,
                tgt_key_padding_mask: Tensor | None = None,
                memory_key_padding_mask: Tensor | None = None) -> Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
        return output


class Transformer(Module):
    def __init__(self, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", batch_first: bool = True) -> None:
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.d_model = d_model

    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Tensor | None = None, tgt_mask: Tensor | None = None,
                memory_mask: Tensor | None = None,
                src_key_padding_mask: Tensor | None = None,
                tgt_key_padding_mask: Tensor | None = None,
                memory_key_padding_mask: Tensor | None = None) -> Tensor:
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output
