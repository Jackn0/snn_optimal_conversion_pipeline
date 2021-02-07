import argparse

parser = argparse.ArgumentParser(description='model parameters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', default='CIFAR100', type=str, help='dataset name',
                    choices=['MNIST', 'CIFAR10', 'CIFAR100'])
parser.add_argument('--arch', default='VGG16', type=str, help='dataset name',
                    choices=['VGG16', 'ResNet20', 'CIFARNet'])
parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='initial learning_rate')
parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--thresh', default=0, type=int, help='relu threshold, 0 means use common relu')
parser.add_argument('--T', default=32, type=int, help='snn simulation length')
parser.add_argument('--shift_relu', default=0, type=int, help='ReLU shift reference time')
parser.add_argument('--shift_snn', default=32, type=int, help='SNN left shift reference time')
parser.add_argument('--step', default=4, type=int, help='record snn output per step')
parser.add_argument('--init_epoch', default=-1, type=int, help='use ulimited relu to init parameters')

args = parser.parse_args()

if __name__ == '__main__':
    print(args.learning_rate)
