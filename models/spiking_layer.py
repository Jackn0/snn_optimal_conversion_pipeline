"""
@author: Shikuang Deng
"""
import torch
import torch.nn as nn
from .settings import *


# spike layer, requires nn.Conv2d (nn.Linear) and thresh
class SPIKE_layer(nn.Module):
    def __init__(self, thresh, Conv2d):
        super(SPIKE_layer, self).__init__()
        self.thresh = thresh
        self.ops = Conv2d
        self.mem = 0
        if args.shift_snn > 0:
            self.shift = self.thresh / (2 * args.shift_snn)
        else:
            self.shift = 0
        # self.shift_operation()

    # Initialize the membrane potential
    def init_mem(self):
        self.mem = 0

    def shift_operation(self):
        for key in self.state_dict().keys():
            if 'bias' in key:
                pa = self.state_dict()[key]
                pa.copy_(pa + self.shift)

    def forward(self, input):
        # spiking
        x = self.ops(input) + self.shift
        self.mem += x
        spike = self.mem.ge(self.thresh).float() * self.thresh
        # soft-rest
        self.mem -= spike
        return spike


# create the fake snn layer, but there is error that can't be reduce, like the membrane increase is
# 0.9 0.3 -2 0.4    snn will fire one spike but, fake snn and ann have no output 
class Fake_layer(nn.Module):
    def __init__(self, thresh, ops, T):
        super(Fake_layer, self).__init__()
        self.ops = ops
        self.thresh = thresh
        self.T = T
        self.shift = self.thresh / (2 * args.shift_snn)

    def forward(self, x):
        x = self.ops(x) + self.shift
        x = x / (self.thresh / self.T)
        x = torch.floor(x)
        x = torch.clamp(x, 0, self.T)
        x = x * (self.thresh / self.T)
        return x
