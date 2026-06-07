from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn.modules import Module, Parameter
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

        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                gs = self._gate_size
                suffix = "_reverse" if direction == 1 else ""
                w_ih = Parameter(torch.randn((gs, layer_input_size)) * 0.01)
                w_hh = Parameter(torch.randn((gs, hidden_size)) * 0.01)
                setattr(self, f"weight_ih_l{layer}{suffix}", w_ih)
                setattr(self, f"weight_hh_l{layer}{suffix}", w_hh)
                if bias:
                    b_ih = Parameter(torch.zeros((gs,)))
                    b_hh = Parameter(torch.zeros((gs,)))
                    setattr(self, f"bias_ih_l{layer}{suffix}", b_ih)
                    setattr(self, f"bias_hh_l{layer}{suffix}", b_hh)

    def _set_gate_size(self) -> None:
        self._gate_size = 4 * self.hidden_size if self.mode in ("LSTM",) else self.hidden_size
        if self.mode == "GRU":
            self._gate_size = 3 * self.hidden_size

    def _param(self, base: str, layer: int, direction: int = 0) -> Tensor:
        suffix = "_reverse" if direction == 1 else ""
        return getattr(self, f"{base}_l{layer}{suffix}")

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

        num_states = self.num_layers * self.num_directions
        if hx is None:
            h_list = [torch.zeros((batch_size, self.hidden_size)) for _ in range(num_states)]
            c_list = [torch.zeros((batch_size, self.hidden_size)) for _ in range(num_states)]
        else:
            h0, c0 = hx
            h_list = [h0[i] for i in range(num_states)]
            c_list = [c0[i] for i in range(num_states)]

        outputs: list[Tensor] = []
        for t in range(seq_len):
            xt = x[t]
            for layer in range(self.num_layers):
                h_f, c_f = self._lstm_step(
                    xt, h_list[layer], c_list[layer],
                    self._param("weight_ih", layer),
                    self._param("weight_hh", layer),
                    self._param("bias_ih", layer) if self.bias else torch.zeros((4 * self.hidden_size,)),
                    self._param("bias_hh", layer) if self.bias else torch.zeros((4 * self.hidden_size,)),
                )
                if self.bidirectional:
                    xt_rev = x[seq_len - 1 - t]
                    h_b, c_b = self._lstm_step(
                        xt_rev, h_list[layer + self.num_layers], c_list[layer + self.num_layers],
                        self._param("weight_ih", layer, 1),
                        self._param("weight_hh", layer, 1),
                        self._param("bias_ih", layer, 1) if self.bias else torch.zeros((4 * self.hidden_size,)),
                        self._param("bias_hh", layer, 1) if self.bias else torch.zeros((4 * self.hidden_size,)),
                    )
                    h_list[layer] = h_f
                    h_list[layer + self.num_layers] = h_b
                    c_list[layer] = c_f
                    c_list[layer + self.num_layers] = c_b
                    xt = torch.cat([h_f, h_b], dim=-1)
                else:
                    h_list[layer] = h_f
                    c_list[layer] = c_f
                    xt = h_f
            outputs.append(xt)

        h = torch.stack(h_list, dim=0)
        c = torch.stack(c_list, dim=0)
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

        num_states = self.num_layers * self.num_directions
        if hx is None:
            h_list = [torch.zeros((batch_size, self.hidden_size)) for _ in range(num_states)]
        else:
            h_list = [hx[i] for i in range(num_states)]

        outputs: list[Tensor] = []
        for t in range(seq_len):
            xt = x[t]
            for layer in range(self.num_layers):
                h_f = self._gru_step(
                    xt, h_list[layer],
                    self._param("weight_ih", layer),
                    self._param("weight_hh", layer),
                    self._param("bias_ih", layer) if self.bias else torch.zeros((3 * self.hidden_size,)),
                    self._param("bias_hh", layer) if self.bias else torch.zeros((3 * self.hidden_size,)),
                )
                if self.bidirectional:
                    xt_rev = x[seq_len - 1 - t]
                    h_b = self._gru_step(
                        xt_rev, h_list[layer + self.num_layers],
                        self._param("weight_ih", layer, 1),
                        self._param("weight_hh", layer, 1),
                        self._param("bias_ih", layer, 1) if self.bias else torch.zeros((3 * self.hidden_size,)),
                        self._param("bias_hh", layer, 1) if self.bias else torch.zeros((3 * self.hidden_size,)),
                    )
                    h_list[layer] = h_f
                    h_list[layer + self.num_layers] = h_b
                    xt = torch.cat([h_f, h_b], dim=-1)
                else:
                    h_list[layer] = h_f
                    xt = h_f
            outputs.append(xt)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(outputs, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
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

        num_states = self.num_layers * self.num_directions
        if hx is None:
            h_list = [torch.zeros((batch_size, self.hidden_size)) for _ in range(num_states)]
        else:
            h_list = [hx[i] for i in range(num_states)]

        step_fn = self._rnn_tanh_step if self.nonlinearity == "tanh" else self._rnn_relu_step

        outputs: list[Tensor] = []
        for t in range(seq_len):
            xt = x[t]
            for layer in range(self.num_layers):
                h_f = step_fn(xt, h_list[layer],
                              self._param("weight_ih", layer),
                              self._param("weight_hh", layer),
                              self._param("bias_ih", layer) if self.bias else torch.zeros((self.hidden_size,)),
                              self._param("bias_hh", layer) if self.bias else torch.zeros((self.hidden_size,)))
                if self.bidirectional:
                    xt_rev = x[seq_len - 1 - t]
                    h_b = step_fn(xt_rev, h_list[layer + self.num_layers],
                                  self._param("weight_ih", layer, 1),
                                  self._param("weight_hh", layer, 1),
                                  self._param("bias_ih", layer, 1) if self.bias else torch.zeros((self.hidden_size,)),
                                  self._param("bias_hh", layer, 1) if self.bias else torch.zeros((self.hidden_size,)))
                    h_list[layer] = h_f
                    h_list[layer + self.num_layers] = h_b
                    xt = torch.cat([h_f, h_b], dim=-1)
                else:
                    h_list[layer] = h_f
                    xt = h_f
            outputs.append(xt)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(outputs, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h
