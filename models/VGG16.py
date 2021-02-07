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


class VGG16(nn.Module):
    def __init__(self, modify):
        super(VGG16, self).__init__()
        # GROUP 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:32*32*64
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:32*32*64
        self.maxpool1 = nn.AvgPool2d(2)
        # GROUP 2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:16*16*128
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:16*16*128
        self.maxpool2 = nn.AvgPool2d(2)
        # GROUP 3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:8*8*256
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:8*8*256
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)  # output:8*8*256
        self.maxpool3 = nn.AvgPool2d(2)
        # GROUP 4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1,
                                 padding=1)  # output:4*4*512
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1)  # output:4*4*512
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)  # output:4*4*512
        self.maxpool4 = nn.AvgPool2d(2)
        # GROUP 5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1)  # output:14*14*512
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1)  # output:14*14*512
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)  # output:14*14*512
        # self.maxpool5 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(in_features=512 * 2 * 2, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        if args.dataset == 'CIFAR100':
            self.fc3 = nn.Linear(in_features=4096, out_features=100)
        else:
            self.fc3 = nn.Linear(in_features=4096, out_features=10)
        self.init_epoch = args.init_epoch
        self.relu = th_shift_ReLU(args.shift_relu, modify)
        self.max_active = [0] * 16

    def renew_max(self, x, y, epoch):
        # if epoch > self.init_epoch:
        x = max(x, y)
        return x

    def forward(self, x, epoch):
        # GROUP 1
        output = self.conv1_1(x)
        output = self.relu(output)
        self.max_active[0] = self.renew_max(self.max_active[0], output.max(), epoch)
        output = self.conv1_2(output)
        output = self.relu(output)
        self.max_active[1] = self.renew_max(self.max_active[1], output.max(), epoch)
        output = self.maxpool1(output)
        # GROUP 2
        output = self.conv2_1(output)
        output = self.relu(output)
        self.max_active[2] = self.renew_max(self.max_active[2], output.max(), epoch)
        output = self.conv2_2(output)
        output = self.relu(output)
        self.max_active[3] = self.renew_max(self.max_active[3], output.max(), epoch)
        output = self.maxpool2(output)
        # GROUP 3
        output = self.conv3_1(output)
        output = self.relu(output)
        self.max_active[4] = self.renew_max(self.max_active[4], output.max(), epoch)
        output = self.conv3_2(output)
        output = self.relu(output)
        self.max_active[5] = self.renew_max(self.max_active[5], output.max(), epoch)
        output = self.conv3_3(output)
        output = self.relu(output)
        self.max_active[6] = self.renew_max(self.max_active[6], output.max(), epoch)
        output = self.maxpool3(output)
        # GROUP 4
        output = self.conv4_1(output)
        output = self.relu(output)
        self.max_active[7] = self.renew_max(self.max_active[7], output.max(), epoch)
        output = self.conv4_2(output)
        output = self.relu(output)
        self.max_active[8] = self.renew_max(self.max_active[8], output.max(), epoch)
        output = self.conv4_3(output)
        output = self.relu(output)
        self.max_active[9] = self.renew_max(self.max_active[9], output.max(), epoch)
        output = self.maxpool4(output)
        # GROUP 5
        output = self.conv5_1(output)
        output = self.relu(output)
        self.max_active[10] = self.renew_max(self.max_active[10], output.max(), epoch)
        output = self.conv5_2(output)
        output = self.relu(output)
        self.max_active[11] = self.renew_max(self.max_active[11], output.max(), epoch)
        output = self.conv5_3(output)
        output = self.relu(output)
        self.max_active[12] = self.renew_max(self.max_active[12], output.max(), epoch)
        # output = self.maxpool5(output)
        output = output.view(x.size(0), -1)
        output = self.fc1(output)
        output = self.relu(output)
        self.max_active[13] = self.renew_max(self.max_active[13], output.max(), epoch)
        output = self.fc2(output)
        output = self.relu(output)
        self.max_active[14] = self.renew_max(self.max_active[14], output.max(), epoch)
        output = self.fc3(output)
        self.max_active[15] = self.renew_max(self.max_active[15], output.max(), epoch)
        return output

    def record(self):
        return np.array(self.max_active)

    def load_max_active(self, mat):
        self.max_active = mat


class VGG16_fake(nn.Module):
    def __init__(self, thresh_list, model):
        super(VGG16_fake, self).__init__()
        # group1
        self.conv1_1 = Fake_layer(thresh_list[0], model.conv1_1, args.T)
        self.conv1_2 = Fake_layer(thresh_list[1], model.conv1_2, args.T)
        self.pool1 = nn.AvgPool2d(2)
        # group2
        self.conv2_1 = Fake_layer(thresh_list[2], model.conv2_1, args.T)
        self.conv2_2 = Fake_layer(thresh_list[3], model.conv2_2, args.T)
        self.pool2 = nn.AvgPool2d(2)
        # group3
        self.conv3_1 = Fake_layer(thresh_list[4], model.conv3_1, args.T)
        self.conv3_2 = Fake_layer(thresh_list[5], model.conv3_2, args.T)
        self.conv3_3 = Fake_layer(thresh_list[6], model.conv3_3, args.T)
        self.pool3 = nn.AvgPool2d(2)
        # group4
        self.conv4_1 = Fake_layer(thresh_list[7], model.conv4_1, args.T)
        self.conv4_2 = Fake_layer(thresh_list[8], model.conv4_2, args.T)
        self.conv4_3 = Fake_layer(thresh_list[9], model.conv4_3, args.T)
        self.pool4 = nn.AvgPool2d(2)
        # group5
        self.conv5_1 = Fake_layer(thresh_list[10], model.conv5_1, args.T)
        self.conv5_2 = Fake_layer(thresh_list[11], model.conv5_2, args.T)
        self.conv5_3 = Fake_layer(thresh_list[12], model.conv5_3, args.T)

        self.fc1 = Fake_layer(thresh_list[13], model.fc1, args.T)
        self.fc2 = Fake_layer(thresh_list[14], model.fc2, args.T)
        self.fc3 = Fake_layer(thresh_list[15], model.fc3, args.T)

    def forward(self, x):
        output = self.conv1_1(x)
        output = self.relu(output)
        output = self.conv1_2(output)
        output = self.relu(output)
        output = self.maxpool1(output)
        output = self.conv2_1(output)
        output = self.relu(output)
        output = self.conv2_2(output)
        output = self.relu(output)
        output = self.maxpool2(output)
        output = self.conv3_1(output)
        output = self.relu(output)
        output = self.conv3_2(output)
        output = self.relu(output)
        output = self.conv3_3(output)
        output = self.relu(output)
        output = self.maxpool3(output)
        output = self.conv4_1(output)
        output = self.relu(output)
        output = self.conv4_2(output)
        output = self.relu(output)
        output = self.conv4_3(output)
        output = self.relu(output)
        output = self.maxpool4(output)
        output = self.conv5_1(output)
        output = self.relu(output)
        output = self.conv5_2(output)
        output = self.relu(output)
        output = self.conv5_3(output)
        output = self.relu(output)
        output = output.view(x.size(0), -1)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output


class VGG16_spiking(nn.Module):
    def __init__(self, thresh_list, model):
        super(VGG16_spiking, self).__init__()
        # group1
        self.conv1_1 = SPIKE_layer(thresh_list[0], model.conv1_1)
        self.conv1_2 = SPIKE_layer(thresh_list[1], model.conv1_2)
        self.pool1 = nn.AvgPool2d(2)
        # group2
        self.conv2_1 = SPIKE_layer(thresh_list[2], model.conv2_1)
        self.conv2_2 = SPIKE_layer(thresh_list[3], model.conv2_2)
        self.pool2 = nn.AvgPool2d(2)
        # group3
        self.conv3_1 = SPIKE_layer(thresh_list[4], model.conv3_1)
        self.conv3_2 = SPIKE_layer(thresh_list[5], model.conv3_2)
        self.conv3_3 = SPIKE_layer(thresh_list[6], model.conv3_3)
        self.pool3 = nn.AvgPool2d(2)
        # group4
        self.conv4_1 = SPIKE_layer(thresh_list[7], model.conv4_1)
        self.conv4_2 = SPIKE_layer(thresh_list[8], model.conv4_2)
        self.conv4_3 = SPIKE_layer(thresh_list[9], model.conv4_3)
        self.pool4 = nn.AvgPool2d(2)
        # group5
        self.conv5_1 = SPIKE_layer(thresh_list[10], model.conv5_1)
        self.conv5_2 = SPIKE_layer(thresh_list[11], model.conv5_2)
        self.conv5_3 = SPIKE_layer(thresh_list[12], model.conv5_3)

        self.fc1 = SPIKE_layer(thresh_list[13], model.fc1)
        self.fc2 = SPIKE_layer(thresh_list[14], model.fc2)
        self.fc3 = SPIKE_layer(thresh_list[15], model.fc3)
        self.T = args.T

    def init_layer(self):
        self.conv1_1.init_mem()
        self.conv1_2.init_mem()
        self.conv2_1.init_mem()
        self.conv2_2.init_mem()
        self.conv3_1.init_mem()
        self.conv3_2.init_mem()
        self.conv3_3.init_mem()
        self.conv4_1.init_mem()
        self.conv4_2.init_mem()
        self.conv4_3.init_mem()
        self.conv5_1.init_mem()
        self.conv5_2.init_mem()
        self.conv5_3.init_mem()
        self.fc1.init_mem()
        self.fc2.init_mem()
        self.fc3.init_mem()

    def forward(self, x):
        self.init_layer()
        with torch.no_grad():
            out_spike_sum = 0
            result_list = []
            for time in range(self.T):
                spike_input = x
                output = self.conv1_1(spike_input)
                output = self.conv1_2(output)
                output = self.pool1(output)
                # group 2
                output = self.conv2_1(output)
                output = self.conv2_2(output)
                output = self.pool2(output)
                # group 3
                output = self.conv3_1(output)
                output = self.conv3_2(output)
                output = self.conv3_3(output)
                output = self.pool3(output)
                # group 4
                output = self.conv4_1(output)
                output = self.conv4_2(output)
                output = self.conv4_3(output)
                output = self.pool4(output)
                # group 5
                output = self.conv5_1(output)
                output = self.conv5_2(output)
                output = self.conv5_3(output)
                #
                output = output.view(x.size(0), -1)
                output = self.fc1(output)
                output = self.fc2(output)
                output = self.fc3(output)
                out_spike_sum += output
                if (time + 1) % args.step == 0:
                    sub_result = out_spike_sum / (time + 1)
                    result_list.append(sub_result)
        return result_list
