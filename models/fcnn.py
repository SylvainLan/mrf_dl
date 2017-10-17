# -*- coding: utf-8 -*-
"""
Fully connected neural net
=========================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, d_in, inner_layer_size=[50, 50], d_out=2):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(d_in, inner_layer_size[0])
        self.hidden_layer = nn.Linear(inner_layer_size[0], inner_layer_size[1])
        self.output_layer = nn.Linear(inner_layer_size[1], d_out)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = F.relu(self.output_layer(x))
        return x
