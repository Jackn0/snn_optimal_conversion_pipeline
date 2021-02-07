"""
@author: Shikuang Deng
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .new_relu import *
from .spiking_layer import *
from .settings import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, modify):
        super(CNN, self).__init__()
        self.relu = th_shift_ReLU(args.shift_relu, modify)
        self.conv1 = nn.Conv2d(1, 3, 5, 1, 2)
        self.conv1_max = -999
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(3, 6, 5, 1, 2)
        self.conv2_max = -999
        self.pool2 = nn.AvgPool2d(2)
        self.fc = nn.Linear(6 * 7 * 7, 10)
        self.fc_max = -999
        self.init_epoch = args.init_epoch

    def forward(self, x, epoch):
        if epoch > self.init_epoch:
            active = self.relu
        else:
            active = F.relu
        x = self.conv1(x)
        x = active(x)
        if epoch > self.init_epoch:
            self.conv1_max = max(x.max().item(), self.conv1_max)
        x = self.pool1(x)
        x = self.conv2(x)
        x = active(x)
        if epoch > self.init_epoch:
            self.conv2_max = max(x.max().item(), self.conv2_max)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if epoch > self.init_epoch:
            self.fc_max = max(x.max().item(), self.fc_max)
        return x

    def record(self):
        max_act = [0] * 3
        max_act[0] = self.conv1_max
        max_act[1] = self.conv2_max
        max_act[2] = self.fc_max
        return np.array(max_act)

    def load_max_active(self, mat):
        self.conv1_max = mat[0]
        self.conv2_max = mat[1]
        self.fc_max = mat[2]


class SpikeCNN(nn.Module):
    # copy CNN's max layer output and operation
    def __init__(self, CNN):
        super(SpikeCNN, self).__init__()
        self.T = args.T
        self.conv1 = SPIKE_layer(CNN.conv1_max, CNN.conv1)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = SPIKE_layer(CNN.conv2_max, CNN.conv2)
        self.pool2 = nn.AvgPool2d(2)
        self.fc = SPIKE_layer(CNN.fc_max, CNN.fc)

    def forward(self, input):
        # no poisson spike trains
        out_spike_num = torch.zeros(input.size(0), 10, device=device)
        result_list = []
        for time in range(self.T):
            s1 = self.conv1(input)
            s1 = self.pool1(s1)
            s2 = self.conv2(s1)
            s2 = self.pool2(s2)
            s2 = s2.view(s2.size(0), -1)
            out = self.fc(s2)
            out_spike_num += out
            if (time + 1) % args.step == 0:
                sub_result = out_spike_num / (time + 1)
                result_list.append(sub_result)
        return result_list

    def init_layer(self):
        self.conv1.init_mem()
        self.conv2.init_mem()
        self.fc.init_mem()


if __name__ == "__main__":
    pass
    # cn = CNN(True)
    # c = CNNRecord()
    # c.renew(cn)
    # c.ccc()
    # print(c.conv1_max)
    # c.save()
    #
    # d = CNNRecord()
    # d.load()
    # print(d.conv1_max)
