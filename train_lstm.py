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
lastL.cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(gf.parameters(), lr=0.01, momentum=0.9)
gf.cuda()

# for epoch in range(1):
#     for i, data in enumerate(trainloader):
#         sig, T = data
#         sig, T = Variable(sig.cuda()), Variable(T.cuda())
#         optimizer.zero_grad()
#         h = [[Variable(torch.zeros(b_size,100).cuda()), Variable(torch.zeros(b_size,100).cuda())]]
#         c = [Variable(torch.zeros(b_size,100).cuda()), Variable(torch.zeros(b_size,100).cuda())]
#         for t in range(11):
#             h_next, c = gf(sig, h[t], c)
#             h.append(h_next)
#         last_out = torch.cat([h[l + 1][-1] for l in range(10)], 1)
#         out = F.sigmoid(lastL(last_out))
#         loss = criterion(out, T)
#         loss.backward()
#         optimizer.step()

test_iter = iter(testloader)
sig_t, T_t = test_iter.next()
sig_t, T_t = Variable(sig_t), Variable(T_t)
out = net(sig_t)
print(out)
print(T_t)
