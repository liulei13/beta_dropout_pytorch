# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.utils as utils
import numpy as np
from torch.autograd import Function
import os
import time
# define my function
# Traning settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default:100)')
parser.add_argument('--test_batch_size', type=int, default=10000, metavar='N',
                    help='input batch size for testing (default:1000)')
parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                    help='number of epochs to train (default:10000)')
parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                    help='learning rate (default:0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default:0.5)')
parser.add_argument('--mu', type=float, default=0.5, metavar='M',
                    help='mean (default:0.5)')
parser.add_argument('--sigma2', type=float, default=0.3, metavar='M',
                    help='sigma2 (default:0.3)')
parser.add_argument('--p', type=float, default=0.5, metavar='M',
                    help='p for bernoulli (default:0.3)')
parser.add_argument('--a', type=float, default=0.0, metavar='M',
                    help='a for uniform (default:0.0)')
parser.add_argument('--b', type=float, default=1.0, metavar='M',
                    help='b for uniform  (default:1.0)')
parser.add_argument('--A', type=float, default=0.5, metavar='M',
                    help='A for beta (default:0.5)')
parser.add_argument('--B', type=float, default=0.5, metavar='M',
                    help='B for beta  (default:0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed (default:1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--init_clip_max_norm', type=int, default=10, metavar='N',
                    help='init_clip_max_norm')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#print('GPU:', args.cuda)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

def print_log(print_string, log_file):
    """打印并写入指定文件.
           Arguments:
               print_string (string): 打印的字符串
               log_file (FILE *):  写入的文件指针

           Returns:
               nothing
    """
    print("{}".format(print_string))
    #
    log_file.write('{}\n'.format(print_string))
    # 刷新缓存区，将数据写入
    log_file.flush()

class Beta_Dropout(Function):
    def __init__(self,A=1.0,B=1.0,train=False, inplace=False):
        super(Beta_Dropout, self).__init__()
        self.A=A
        self.B=B
        self.train = train
        self.inplace = inplace

    def _make_noise(self, input):
        return input.new().resize_as_(input)

    def forward(self, input):
        if self.inplace:
            self.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        if  self.train:
            #self.noise = self._make_noise(input)
            #self.noise.bernoulli_(1 - self.p).div_(1 - self.p)
            self.noise = np.random.beta(self.A, self.B, size=input.size()) / (self.A / (self.A + self.B))
            self.noise = torch.Tensor(self.noise)
            if input.is_cuda:
                self.noise = self.noise.cuda()
            #self.noise= sample
            self.noise = self.noise.expand_as(input)
            output.mul_(self.noise)
        return output

    def backward(self, grad_output):
        if self.train:
            return grad_output.mul(self.noise)
        else:
            return grad_output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(28 * 28, 800),
        )
        self.relu1=nn.ReLU(inplace=True)
        self.fc2 = nn.Sequential(
            nn.Linear(800, 800),)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Sequential(
            nn.Linear(800, 10) )
        self.dropout_layer1 = Beta_Dropout(args.B, args.B, self.training, False)
        self.dropout_layer2 = Beta_Dropout(args.B, args.B, self.training, False)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, torch.sqrt(2. / n))
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, input):
        output = self.relu1(self.fc1(input.view(-1,28*28)))
        output = self.dropout_layer1(output)
        output=self.relu2(self.fc2(output))
        output = self.dropout_layer2(output)
        output = self.fc3(output)
        output = F.log_softmax(output,dim=1)
        return output

model = Net()
model_path='mlp_adaptive_beta_dropout_B_0.5_lr_0.01_tp_1500_88.pth'
model=torch.load(model_path)
if args.cuda:
    model.cuda()

def test():
    model.eval()
    test_loss = 0
    correct = 0
    model.dropout_layer1 = Beta_Dropout(args.B, args.B, False, False)
    model.dropout_layer2 = Beta_Dropout(args.B, args.B, False, False)
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Error: {}'.format(test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), len(test_loader.dataset) - correct))

if __name__=='__main__':
    epoch_time_start=time.time()
    test()
    print('cost time for epoch:{}'.format(time.time()-epoch_time_start))

