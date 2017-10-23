# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from models.gflstm import GFLSTM
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data.mag_data_load import mag_load

b_size = 4

trainset = mag_load('./data/T1_class', train=True, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True)
testset = mag_load('./data/T1_class', train=False, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

gf = GFLSTM(850, 100, 2)
lastL = nn.Linear(1000, 44)
criterion = nn.MSELoss()
optimizer = optim.SGD(gf.parameters(), lr=0.01, momentum=0.9)
gf.cuda()