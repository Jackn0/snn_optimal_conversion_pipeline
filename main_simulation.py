import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import time
from models import *
from main_train import dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = args.batch_size
activation_save_name = args.arch + '_' + args.dataset + '_activation.npy'
model_save_name = args.arch + '_' + args.dataset + '_state_dict.pth'

data_path = './raw/'  # dataset path

if __name__ == '__main__':
    train_loader, test_loader = dataset()

    if args.thresh > 0:
        relu_th = True
    else:
        relu_th = False
    # use threshold ReLU
    if args.arch == 'VGG16':
        ann = VGG16(relu_th)
    elif args.arch == 'ResNet20':
        ann = ResNet20(relu_th)
    else:
        ann = CIFARNet(relu_th)
    ann.to(device)
    if torch.cuda.is_available():
        pretrained_dict = torch.load(model_save_name)  # ,map_location=torch.device('cuda'))
    else:
        pretrained_dict = torch.load(model_save_name, map_location=torch.device('cpu'))
    ann.load_state_dict(pretrained_dict)

    if args.arch == 'ResNet20':
        ann = ResNet20x(ann)

    correct = 0
    total = 0

    # # validate on test set with ANN.
    # with torch.no_grad():
    #     for batch_idx, (inputs, targets) in enumerate(test_loader):
    #         ann.eval()
    #         inputs = inputs.to(device)
    #         targets.to(device)
    #         outputs = ann(inputs, 90)
    #         _, predicted = outputs.cpu().max(1)
    #         total += float(targets.size(0))
    #         correct += float(predicted.eq(targets).sum().item())
    #         if batch_idx % 100 == 0:
    #             acc = 100. * float(correct) / float(total)
    #             print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    #         # break
    # print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))

    # find the maximum activation on training set
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            ann.eval()
            inputs = inputs.to(device)
            targets.to(device)
            outputs = ann(inputs, 90)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            if batch_idx % 100 == 0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
            # break
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))

    if args.arch == 'VGG16':
        print(ann.max_active)
        snn = VGG16_spiking(ann.max_active, ann)
    elif args.arch == 'ResNet20':
        snn = ResNet20spike(ann)
    else:
        snn = CIFARNet_spiking(ann.thresh_list, ann)
    snn.to(device)

    correct = 0
    total = 0

    simulation_length = int(args.T / args.step)
    simulation_loader = torch.zeros(1, simulation_length)

    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            snn.init_layer()
            inputs = inputs.to(device)
            outputs_ls = snn(inputs)
            # targets.to(device)
            total += float(targets.cpu().size(0))
            for i in range(simulation_length):
                _, predicted = outputs_ls[i].cpu().max(1)
                simulation_loader[0, i] += float(predicted.eq(targets).sum().item())
            end_time = time.time()
            if (batch_idx + 1) % 20 == 0:
                end_time = time.time()
                print('Step [%d/%d] , time: %.3f s' % (
                    (batch_idx + 1, 10000 // batch_size, end_time - start_time)))
        corr = 100.000 * simulation_loader / total
        Ts = 0

        np.save('simulation_result.npy', corr)
        for i in range(simulation_length):
            Ts = Ts + args.step
            print('simulation length: ', Ts, ' -> corr: ', corr[0, i].data)
