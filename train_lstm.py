{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from torch.autograd import Variable\n",
    "from models.gflstm_3 import gfLSTM\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "gf = gfLSTM(50, 20, 2)\n",
    "\n",
    "lastL = nn.Linear(200, 5)\n",
    "\n",
    "criterion = nn.MSELoss()\n",
    "optimizer = optim.SGD(gf.parameters(),lr = 0.01, momentum = 0.9)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = Variable(torch.rand(1,1,50))\n",
    "\n",
    "h = [[Variable(torch.zeros(1,1,20)), Variable(torch.zeros(1,1,20))]]\n",
    "c = Variable(torch.zeros(2,1,20))\n",
    "\n",
    "for i in range(10):\n",
    "    h_next, c = gf(y, h[i], c)\n",
    "    h.append(h_next)\n",
    "\n",
    "h_in = torch.cat([h[l + 1][-1] for l in range(10)])\n",
    "output = lastL(h_in.view(1,200))\n",
    "\n",
    "target = Variable(torch.Tensor([0,0,1,0,0]))\n",
    "\n",
    "loss = criterion(output, target)\n",
    "loss.backward()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
