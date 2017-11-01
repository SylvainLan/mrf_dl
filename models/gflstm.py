# -*- coding: utf-8 -*-
"""
Gated feedback LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as func


class AttrProxy(object):
    """Translates index lookups into attribute lookups."""

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class GFLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, steps=1, out_dim=44, rep_time=10):
        super(GFLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.out_dim = out_dim
        self.rep_time = rep_time
        for i in range(steps):
            if i == 0:
                self.add_module('i2h_f_' + str(i), nn.Linear(input_dim, hidden_dim))
                self.add_module('i2h_i_' + str(i), nn.Linear(input_dim, hidden_dim))
                self.add_module('i2h_o_' + str(i), nn.Linear(input_dim, hidden_dim))
                self.add_module('i2h_g_' + str(i), nn.Linear(input_dim, steps))
                self.add_module('i2h_c_' + str(i), nn.Linear(input_dim, hidden_dim))
            else:
                self.add_module('i2h_f_' + str(i), nn.Linear(hidden_dim, hidden_dim))
                self.add_module('i2h_i_' + str(i), nn.Linear(hidden_dim, hidden_dim))
                self.add_module('i2h_o_' + str(i), nn.Linear(hidden_dim, hidden_dim))
                self.add_module('i2h_g_' + str(i), nn.Linear(hidden_dim, steps))
                self.add_module('i2h_c_' + str(i), nn.Linear(hidden_dim, hidden_dim))

            self.add_module('h2h_f_' + str(i), nn.Linear(hidden_dim, hidden_dim))
            self.add_module('h2h_i_' + str(i), nn.Linear(hidden_dim, hidden_dim))
            self.add_module('h2h_o_' + str(i), nn.Linear(hidden_dim, hidden_dim))
            self.add_module('h2h_g_' + str(i), nn.Linear(steps * hidden_dim, steps))
            self.add_module('h2h_c_' + str(i), nn.Linear(steps * hidden_dim, steps * hidden_dim))

        self.i2h_f = AttrProxy(self, 'i2h_f_')
        self.h2h_f = AttrProxy(self, 'h2h_f_')
        self.i2h_i = AttrProxy(self, 'i2h_i_')
        self.h2h_i = AttrProxy(self, 'h2h_i_')
        self.i2h_o = AttrProxy(self, 'i2h_o_')
        self.h2h_o = AttrProxy(self, 'h2h_o_')
        self.i2h_g = AttrProxy(self, 'i2h_g_')
        self.h2h_g = AttrProxy(self, 'h2h_g_')
        self.i2h_c = AttrProxy(self, 'i2h_c_')
        self.h2h_c = AttrProxy(self, 'h2h_c_')
        self.last_l = nn.Linear((rep_time - 1) * hidden_dim, out_dim)

    def forward(self, input, hidden, current):
        for t in range(self.rep_time):
            h_next = []
            c_next = []
            input_h = input
            for l in range(self.steps):
                f = func.sigmoid(self.i2h_f[l](input_h) + self.h2h_f[l](hidden[t][l]))
                i = func.sigmoid(self.i2h_i[l](input_h) + self.h2h_i[l](hidden[t][l]))
                g = func.sigmoid(self.i2h_g[l](input_h) + self.h2h_g[l](torch.cat(hidden[t], 1)))
                aux = self.h2h_c[l](torch.cat(hidden[t], 1))
                aux = aux.view(-1, self.steps, self.hidden_dim)
                g = g.view(-1, self.steps, 1)
                aux = aux * g
                aux = aux.sum(1).view(-1, self.hidden_dim)

                c_t = func.tanh(self.i2h_c[l](input_h) + aux)
                c = f * current[l] + i * c_t
                o = func.sigmoid(self.i2h_o[l](input_h) + self.h2h_o[l](hidden[t][l]))
                h = o * c
                input_h = h
                h_next.append(h)
                c_next.append(c)
            hidden.append(h_next)
            current = c_next
        out = func.sigmoid(self.last_l(torch.cat([hidden[l + 1][-1] for l in range(self.rep_time - 1)], 1)))
        return out
