# -*- coding: utf-8 -*-

from models.fcnn import Net
from data.mag_data_load import mag_load

import torch.optim as optim
import torch
import torch.nn as nn
from torch.autograd import Variable

trainset = mag_load('./data/T1_T2', train=True, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testset = mag_load('./data/T1_T2', train=False, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

net = Net(850, [50, 80], 2)
criterion = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(100):
    running_loss = 0.0;

    for i, data in enumerate(trainloader, 0):
        sig, T = data
        sig, T = Variable(sig), Variable(T)

        optimizer.zero_grad()

        outputs = net(sig)

        loss = criterion(outputs, T)
        loss.backward()
        optimizer.step()

print('done')

test_iter = iter(testloader)
sig_t, T_t = test_iter.next()
sig_t, T_t = Variable(sig_t), Variable(T_t)
out = net(sig_t)
print(out)
print(T_t)

m = 0
for i, data in enumerate(testloader, 0):
    sig, T = data
    sig, T = Variable(sig), Variable(T)
    outputs = net(sig)
    loss = criterion(outputs, T)
    m += loss

m /= i
print(m)
