import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import time
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = args.batch_size
learning_rate = args.learning_rate
num_epochs = args.epochs
data_path = './raw/'
activation_save_name = args.arch + '_' + args.dataset + '_activation.npy'
model_save_name = args.arch + '_' + args.dataset + '_state_dict.pth'

def dataset():
    if args.dataset == 'MNIST':
        trans_train = trans_test = transforms.Compose([transforms.ToTensor(), ])
        train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True,
                                                   transform=trans_train)
        test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True,
                                              transform=transforms.ToTensor())

    elif args.dataset == 'CIFAR10':
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True,
                                                     transform=trans_train)
        test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True,
                                                transform=trans_test)
    elif args.dataset == 'CIFAR100':
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True,
                                                      transform=trans_train)
        test_set = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True,
                                                 transform=trans_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

def lr_scheduler(optimizer, epoch):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr_list = [100, 140, 240]
    if epoch in lr_list:
        print('change the learning rate')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


best_acc = 0
best_epoch = 0

if args.arch == 'CNN2':
    ann = CNN(True)
elif args.arch == 'CIFARNet':
    pass
elif args.arch == 'VGG16':
    pass
elif args.arch == 'ResNet20':
    pass
else:
    print('Unkonwn network')
    ann = CNN(True)

ann.to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(ann.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(ann.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
if __name__ == "__main__":
    train_loader, test_loader = dataset()
    for epoch in range(num_epochs):
        running_loss = 0
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            ann.train()
            ann.zero_grad()
            optimizer.zero_grad()
            labels.to(device)
            images = images.float().to(device)
            outputs = ann(images, epoch)
            loss = criterion(outputs.cpu(), labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, running_loss))
                running_loss = 0
                print('Time elasped:', time.time() - start_time)
        correct = 0
        total = 0
        optimizer = lr_scheduler(optimizer, epoch)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                ann.eval()
                inputs = inputs.to(device)
                optimizer.zero_grad()
                targets.to(device)
                outputs = ann(inputs, epoch)
                loss = criterion(outputs.cpu(), targets)
                _, predicted = outputs.cpu().max(1)
                total += float(targets.size(0))
                correct += float(predicted.eq(targets).sum().item())
                if batch_idx % 100 == 0:
                    acc = 100. * float(correct) / float(total)
                    print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)

        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))

        acc = 100. * float(correct) / float(total)
        if best_acc < acc and epoch > args.init_epoch:
            best_acc = acc
            best_epoch = epoch + 1
            torch.save(ann.state_dict(), model_save_name)
            best_max_act = ann.record()
            # np.save(activation_save_name, best_max_act)
        print('best_acc is: ', best_acc, ' find in epoch: ', best_epoch)
        print('Iters:', epoch, '\n\n\n')
