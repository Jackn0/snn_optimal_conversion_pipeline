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


class CIFARNet(nn.Module):
    def __init__(self, modify):
        super(CIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 256, 3, 1, 1)
        self.pool1 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(256, 512, 3, 1, 1)
        self.pool2 = nn.AvgPool2d(2)
        self.conv4 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv5 = nn.Conv2d(1024, 512, 3, 1, 1)
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        if args.dataset == 'CIFAR100':
            self.fc3 = nn.Linear(512, 100)
        else:
            self.fc3 = nn.Linear(512, 10)
        self.relu = th_shift_ReLU(args.shift_relu, modify)
        self.thresh_list = [0] * 8

    def forward(self, input, epoch):
        x = self.conv1(input)
        x = self.relu(x)
        self.thresh_list[0] = max(self.thresh_list[0], x.max())
        x = self.conv2(x)
        x = self.relu(x)
        self.thresh_list[1] = max(self.thresh_list[1], x.max())
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        self.thresh_list[2] = max(self.thresh_list[2], x.max())
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.relu(x)
        self.thresh_list[3] = max(self.thresh_list[3], x.max())
        x = self.conv5(x)
        x = self.relu(x)
        self.thresh_list[4] = max(self.thresh_list[4], x.max())
        x = x.view(input.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        self.thresh_list[5] = max(self.thresh_list[5], x.max())
        x = self.fc2(x)
        x = self.relu(x)
        self.thresh_list[6] = max(self.thresh_list[6], x.max())
        x = self.fc3(x)
        self.thresh_list[7] = max(self.thresh_list[7], x.max())
        return x


class CIFARNet_spiking(nn.Module):
    def __init__(self, thresh_list, model):
        super(CIFARNet_spiking, self).__init__()
        self.conv1 = SPIKE_layer(thresh_list[0], model.conv1)
        self.conv2 = SPIKE_layer(thresh_list[1], model.conv2)
        self.pool1 = nn.AvgPool2d(2)
        self.conv3 = SPIKE_layer(thresh_list[2], model.conv3)
        self.pool2 = nn.AvgPool2d(2)
        self.conv4 = SPIKE_layer(thresh_list[3], model.conv4)
        self.conv5 = SPIKE_layer(thresh_list[4], model.conv5)
        self.fc1 = SPIKE_layer(thresh_list[5], model.fc1)
        self.fc2 = SPIKE_layer(thresh_list[6], model.fc2)
        self.fc3 = SPIKE_layer(thresh_list[7], model.fc3)

    def forward(self, input):
        T = args.T
        step = args.step
        with torch.no_grad():
            out_spike_sum = torch.zeros(input.size(0), 10, device=device)
            result_list = []
            for time in range(T):
                x = self.conv1(input)
                x = self.conv2(x)
                x = self.pool1(x)

                x = self.conv3(x)
                x = self.pool2(x)
                x = self.conv4(x)
                x = self.conv5(x)
                x = x.view(input.size(0), -1)

                x = self.fc1(x)
                x = self.fc2(x)
                x = self.fc3(x)
                out_spike_sum += x
                if (time + 1) % step == 0:
                    sub_result = out_spike_sum / (time + 1)
                    result_list.append(sub_result)
        return result_list

    def init_layer(self):
        self.conv1.init_mem()
        self.conv2.init_mem()
        self.conv3.init_mem()
        self.conv4.init_mem()
        self.conv5.init_mem()
        self.fc1.init_mem()
        self.fc2.init_mem()
        self.fc3.init_mem()
