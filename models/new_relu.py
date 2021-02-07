"""
@author: Shikuang Deng
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .settings import *

thresh = args.thresh


# ReLU with a threshold
class thReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        tmp = abs(input - thresh / 2) <= thresh / 2
        tmp2 = (input - thresh) > 0
        return input * tmp.float() + tmp2.float() * thresh

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh / 2) <= thresh / 2
        return grad_input * temp.float()


act_fun = thReLU.apply


# choosing normal ReLU or modified ReLU with threshold and right shift
class th_shift_ReLU(nn.Module):
    def __init__(self, simulation_length, modify):
        super(th_shift_ReLU, self).__init__()
        if modify == False:
            self.act = F.relu
        else:
            self.act = act_fun
        self.simulation_length = simulation_length

    def forward(self, input):
        if self.simulation_length == 0:
            return self.act(input)
        else:
            return self.act(input - thresh / (self.simulation_length * 2))


if __name__ == '__main__':
    xrelu = th_shift_ReLU(10, True)
    z = torch.rand(3, 3) * 4
    y = xrelu(z)
    print(y)
