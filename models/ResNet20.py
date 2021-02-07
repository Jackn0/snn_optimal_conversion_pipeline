"""
@author: Shikuang Deng
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .new_relu import *
from .spiking_layer import *
from .settings import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.thresh > 0:
    relu_th = True
else:
    relu_th = False

xrelu = th_shift_ReLU(args.shift_relu, relu_th)


class BasicBlockx(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, ops_list):
        super().__init__()
        self.MAC1 = 0
        self.MAC2 = 0
        self.residual1 = ops_list.residual[0]
        self.ReLU = ops_list.residual[1]
        self.residual2 = ops_list.residual[3]
        self.identity = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.identity = ops_list.identity[0]

    def M_update(self, x1, x2):
        self.MAC1 = max(self.MAC1, x1.item())
        self.MAC2 = max(self.MAC2, x2.item())

    def forward(self, x):
        y = self.residual1(x)
        y = self.ReLU(y)
        z = self.residual2(y)
        w = self.identity(x)
        out = z + w
        out = self.ReLU(out)
        self.M_update(y.max(), out.max())
        return out


class ResNetx(nn.Module):
    def __init__(self, block, num_blocks, labels, ops_list):
        super(ResNetx, self).__init__()
        self.in_planes = 64
        self.pre_process1 = ops_list.pre_process[0]
        self.pre_process2 = ops_list.pre_process[3]
        self.pre_process3 = ops_list.pre_process[6]
        self.pre_pool = nn.AvgPool2d(2)
        self.ReLU = ops_list.pre_process[1]
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, ops_list.layer1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, ops_list.layer2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, ops_list.layer3)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, ops_list.layer4)
        self.classifier = ops_list.classifier[0]
        self.MACpre = [-99, -99, -99]
        self.classifier_MAC = -999

    def _make_layer(self, block, planes, num_blocks, stride, ops_list):
        if num_blocks == 0:
            return nn.Sequential()
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        count = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, ops_list[count]))
            self.in_planes = planes * block.expansion
            count += 1
        return nn.Sequential(*layers)

    def new_max_active(self, p1, p2, p3):
        self.MACpre[0] = max(self.MACpre[0], p1.item())
        self.MACpre[1] = max(self.MACpre[1], p2.item())
        self.MACpre[2] = max(self.MACpre[2], p3.item())

    def forward(self, x, epoch):
        out = self.pre_process1(x)
        out1 = self.ReLU(out)
        out = self.pre_process2(out1)
        out2 = self.ReLU(out)
        out = self.pre_process3(out2)
        out3 = self.ReLU(out)
        self.new_max_active(out1.max(), out2.max(), out3.max())
        out3 = self.pre_pool(out3)
        outl1 = self.layer1(out3)
        outl2 = self.layer2(outl1)
        outl3 = self.layer3(outl2)
        outl4 = self.layer4(outl3)
        out = outl4.view(x.size(0), -1)
        out = self.classifier(out)
        self.classifier_MAC = max(self.classifier_MAC, out.max().item())
        return out


class BasicBlockSpiking(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, ops_list):
        super().__init__()
        self.residual1 = SPIKE_layer(ops_list.MAC1, ops_list.residual1)
        self.residual2 = ops_list.residual2
        self.identity = nn.Sequential()
        self.Add_mem = 0
        self.Add_thresh = ops_list.MAC2
        if stride != 1 or in_planes != self.expansion * planes:
            self.identity = ops_list.identity

    def init_mem(self):
        self.Add_mem = 0
        self.residual1.init_mem()

    def forward(self, x):
        y = self.residual1(x)
        mem_up1 = self.residual2(y)
        mem_up2 = self.identity(x)
        self.Add_mem += (mem_up1 + mem_up2)  #
        out_spike = self.Add_mem.ge(self.Add_thresh).float() * self.Add_thresh
        self.Add_mem -= out_spike  # * self.Add_thresh
        # out_spike = xrelu(self.Add_mem)
        return out_spike


class ResNetSpiking(nn.Module):
    def __init__(self, block, num_blocks, labels, ops_list):
        super(ResNetSpiking, self).__init__()
        self.in_planes = 64
        self.pre_process1 = SPIKE_layer(ops_list.MACpre[0], ops_list.pre_process1)
        self.pre_process2 = SPIKE_layer(ops_list.MACpre[1], ops_list.pre_process2)
        self.pre_process3 = SPIKE_layer(ops_list.MACpre[2], ops_list.pre_process3)
        self.pre_pool = ops_list.pre_pool
        self.layer1 = self.__make_layer(block, 64, num_blocks[0], 1, ops_list.layer1)
        self.layer2 = self.__make_layer(block, 128, num_blocks[1], 2, ops_list.layer2)
        self.layer3 = self.__make_layer(block, 256, num_blocks[2], 2, ops_list.layer3)
        self.layer4 = self.__make_layer(block, 512, num_blocks[3], 2, ops_list.layer4)
        self.classifier = SPIKE_layer(ops_list.classifier_MAC, ops_list.classifier)

    def __make_layer(self, block, planes, num_blocks, stride, ops_list):
        if num_blocks == 0:
            return nn.Sequential()
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        count = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, ops_list[count]))
            self.in_planes = planes * block.expansion
            count += 1
        return nn.Sequential(*layers)

    def init_layer(self):
        self.pre_process1.init_mem()
        self.pre_process2.init_mem()
        self.pre_process3.init_mem()
        for i in range(2):
            self.layer1[i].init_mem()
            self.layer2[i].init_mem()
            self.layer3[i].init_mem()
            self.layer4[i].init_mem()
        self.classifier.init_mem()

    def forward(self, x):
        T = args.T
        step = args.step
        with torch.no_grad():
            out_spike_sum = torch.zeros(x.size(0), 10, device=device)
            result_list = []
            for time in range(T):
                out = self.pre_process1(x)
                out = self.pre_process2(out)
                out = self.pre_process3(out)
                out = self.pre_pool(out)
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = out.view(x.size(0), -1)
                out = self.classifier(out)
                out_spike_sum += out
                if (time + 1) % step == 0:
                    sub_result = out_spike_sum / (time + 1)
                    result_list.append(sub_result)
            return result_list


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, dropout):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            xrelu,
            nn.Dropout(dropout),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.identity = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.identity = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),

            )

    def forward(self, x):
        out = self.residual(x) + self.identity(x)
        # out = F.relu(out)
        out = xrelu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, labels=10, dropout=0.2):

        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dropout = dropout
        self.pre_process = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            xrelu,
            nn.Dropout(self.dropout),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            xrelu,
            nn.Dropout(self.dropout),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            xrelu,
            nn.AvgPool2d(2)
        )

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout=self.dropout)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout=self.dropout)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout=self.dropout)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout=self.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, labels, bias=False)
        )
        self._initialize_weights2()

    def _initialize_weights2(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride, dropout):

        if num_blocks == 0:
            return nn.Sequential()
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, e):

        out = self.pre_process(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(x.size(0), -1)
        out = self.classifier(out)
        return out


def ResNet20(dropout=0.2):
    if args.dataset == 'CIFAR100':
        return ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], labels=100, dropout=dropout)
    else:
        return ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], labels=10, dropout=dropout)


def ResNet20x(ops_list):
    if args.dataset == 'CIFAR100':
        return ResNetx(block=BasicBlockx, num_blocks=[2, 2, 2, 2], labels=100, ops_list=ops_list)
    else:
        return ResNetx(block=BasicBlockx, num_blocks=[2, 2, 2, 2], labels=10, ops_list=ops_list)


def ResNet20spike(ops_list):
    if args.dataset == 'CIFAR100':
        return ResNetSpiking(BasicBlockSpiking, [2, 2, 2, 2], 100, ops_list)
    else:
        return ResNetSpiking(BasicBlockSpiking, [2, 2, 2, 2], 10, ops_list)
