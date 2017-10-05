# -*- coding: utf-8 -*-
"""
Fully connected neural net
=========================
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

	def __init__(self, D_in, inner_layer_size = [50,50], D_out=2):
		super(Net,self).__init__()
		self.input_layer = nn.Linear(D_in, inner_layer_size[0])
		self.hidden_layer = nn.Linear(inner_layer_size[0], inner_layer_size[1])
		self.output_layer = nn.Linear(inner_layer_size[1], D_out)
		

	def forward(self,x):
		x = F.relu(self.input_layer(x))
		x = F.relu(self.hidden_layer(x))
		x = F.relu(self.output_layer(x))
		return x

