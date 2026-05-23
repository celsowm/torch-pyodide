from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn.modules import Module
from torch.tensor_factories_ops import tensor_from_data


class RNNBase(Module):
    def __init__(self, mode: str, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, batch_first: bool = True,
                 dropout: float = 0.0, bidirectional: bool = False) -> None:
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self._gate_size: int
        self._set_gate_size()

        self.weight_ih_l: list[Tensor] = []
        self.weight_hh_l: list[Tensor] = []
        self.bias_ih_l: list[Tensor] = []
        self.bias_hh_l: list[Tensor] = []

        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                gs = self._gate_size
                w_ih = torch.randn((gs, layer_input_size)) * 0.01
                w_hh = torch.randn((gs, hidden_size)) * 0.01
                b_ih = torch.zeros((gs,)) if bias else None
                b_hh = torch.zeros((gs,)) if bias else None
                self.weight_ih_l.append(w_ih)
                self.weight_hh_l.append(w_hh)
                if bias:
                    self.bias_ih_l.append(b_ih)
                    self.bias_hh_l.append(b_hh)

    def _set_gate_size(self) -> None:
        self._gate_size = 4 * self.hidden_size if self.mode in ("LSTM",) else self.hidden_size
        if self.mode == "GRU":
            self._gate_size = 3 * self.hidden_size

    def _lstm_step(self, x: Tensor, h: Tensor, c: Tensor,
                   w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> tuple[Tensor, Tensor]:
        gates = x.matmul(w_ih.T) + b_ih + h.matmul(w_hh.T) + b_hh
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def _gru_step(self, x: Tensor, h: Tensor,
                  w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> Tensor:
        gates_x = x.matmul(w_ih.T) + b_ih
        gates_h = h.matmul(w_hh.T) + b_hh
        r_z = gates_x[:, :2 * self.hidden_size] + gates_h[:, :2 * self.hidden_size]
        r = torch.sigmoid(r_z[:, :self.hidden_size])
        z = torch.sigmoid(r_z[:, self.hidden_size:])
        n = torch.tanh(gates_x[:, 2 * self.hidden_size:] + r * gates_h[:, 2 * self.hidden_size:])
        h_new = (1.0 - z) * n + z * h
        return h_new

    def _rnn_relu_step(self, x: Tensor, h: Tensor,
                       w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> Tensor:
        return (x.matmul(w_ih.T) + b_ih + h.matmul(w_hh.T) + b_hh).relu()

    def _rnn_tanh_step(self, x: Tensor, h: Tensor,
                       w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> Tensor:
        return (x.matmul(w_ih.T) + b_ih + h.matmul(w_hh.T) + b_hh).tanh()


class LSTM(RNNBase):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("mode", None)
        super().__init__("LSTM", *args, **kwargs)

    def forward(self, x: Tensor, hx: tuple[Tensor, Tensor] | None = None
                ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.shape

        if hx is None:
            h = torch.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size))
            c = torch.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size))
        else:
            h, c = hx

        outputs: list[Tensor] = []
        for t in range(seq_len):
            xt = x[t]
            for layer in range(self.num_layers):
                idx_f = layer * self.num_directions
                h_f, c_f = self._lstm_step(
                    xt, h[idx_f], c[idx_f],
                    self.weight_ih_l[idx_f], self.weight_hh_l[idx_f],
                    self.bias_ih_l[idx_f] if self.bias else torch.zeros((4 * self.hidden_size,)),
                    self.bias_hh_l[idx_f] if self.bias else torch.zeros((4 * self.hidden_size,)),
                )
                if self.bidirectional:
                    idx_b = idx_f + 1
                    xt_rev = x[seq_len - 1 - t]
                    h_b, c_b = self._lstm_step(
                        xt_rev, h[idx_b], c[idx_b],
                        self.weight_ih_l[idx_b], self.weight_hh_l[idx_b],
                        self.bias_ih_l[idx_b] if self.bias else torch.zeros((4 * self.hidden_size,)),
                        self.bias_hh_l[idx_b] if self.bias else torch.zeros((4 * self.hidden_size,)),
                    )
                    h[idx_f] = h_f
                    h[idx_b] = h_b
                    c[idx_f] = c_f
                    c[idx_b] = c_b
                    xt = torch.cat([h_f, h_b], dim=-1)
                else:
                    h[idx_f] = h_f
                    c[idx_f] = c_f
                    xt = h_f
            outputs.append(xt)

        if self.num_directions == 2:
            for layer in range(1, self.num_layers):
                idx_f = layer * self.num_directions
                idx_b = idx_f + 1
                combined = torch.cat([h[idx_f], h[idx_b]], dim=-1) if self.num_directions == 2 else h[idx_f]
                h[idx_f] = combined[:, :self.hidden_size]
                h[idx_b] = combined[:, self.hidden_size:]

        output = torch.stack(outputs, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, (h, c)


class GRU(RNNBase):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("mode", None)
        super().__init__("GRU", *args, **kwargs)

    def forward(self, x: Tensor, hx: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.shape

        if hx is None:
            h = torch.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size))
        else:
            h = hx

        outputs: list[Tensor] = []
        for t in range(seq_len):
            xt = x[t]
            for layer in range(self.num_layers):
                idx_f = layer * self.num_directions
                h_f = self._gru_step(
                    xt, h[idx_f],
                    self.weight_ih_l[idx_f], self.weight_hh_l[idx_f],
                    self.bias_ih_l[idx_f] if self.bias else torch.zeros((3 * self.hidden_size,)),
                    self.bias_hh_l[idx_f] if self.bias else torch.zeros((3 * self.hidden_size,)),
                )
                if self.bidirectional:
                    idx_b = idx_f + 1
                    xt_rev = x[seq_len - 1 - t]
                    h_b = self._gru_step(
                        xt_rev, h[idx_b],
                        self.weight_ih_l[idx_b], self.weight_hh_l[idx_b],
                        self.bias_ih_l[idx_b] if self.bias else torch.zeros((3 * self.hidden_size,)),
                        self.bias_hh_l[idx_b] if self.bias else torch.zeros((3 * self.hidden_size,)),
                    )
                    h[idx_f] = h_f
                    h[idx_b] = h_b
                    xt = torch.cat([h_f, h_b], dim=-1)
                else:
                    h[idx_f] = h_f
                    xt = h_f
            outputs.append(xt)

        output = torch.stack(outputs, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
        h = h.reshape(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        return output, h


class RNN(RNNBase):
    def __init__(self, *args, nonlinearity: str = "tanh", **kwargs) -> None:
        kwargs.pop("mode", None)
        super().__init__(nonlinearity, *args, **kwargs)
        self.nonlinearity = nonlinearity

    def forward(self, x: Tensor, hx: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.shape

        if hx is None:
            h = torch.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size))
        else:
            h = hx

        step_fn = self._rnn_tanh_step if self.nonlinearity == "tanh" else self._rnn_relu_step

        outputs: list[Tensor] = []
        for t in range(seq_len):
            xt = x[t]
            for layer in range(self.num_layers):
                idx_f = layer * self.num_directions
                h_f = step_fn(xt, h[idx_f],
                              self.weight_ih_l[idx_f], self.weight_hh_l[idx_f],
                              self.bias_ih_l[idx_f] if self.bias else torch.zeros((self.hidden_size,)),
                              self.bias_hh_l[idx_f] if self.bias else torch.zeros((self.hidden_size,)))
                if self.bidirectional:
                    idx_b = idx_f + 1
                    xt_rev = x[seq_len - 1 - t]
                    h_b = step_fn(xt_rev, h[idx_b],
                                  self.weight_ih_l[idx_b], self.weight_hh_l[idx_b],
                                  self.bias_ih_l[idx_b] if self.bias else torch.zeros((self.hidden_size,)),
                                  self.bias_hh_l[idx_b] if self.bias else torch.zeros((self.hidden_size,)))
                    h[idx_f] = h_f
                    h[idx_b] = h_b
                    xt = torch.cat([h_f, h_b], dim=-1)
                else:
                    h[idx_f] = h_f
                    xt = h_f
            outputs.append(xt)

        output = torch.stack(outputs, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h
