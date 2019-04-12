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
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
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
parser.add_argument('--A', type=float, default=3.0, metavar='M',
                    help='A for beta (default:0.5)')
parser.add_argument('--B', type=float, default=3.0, metavar='M',
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

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

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

class UniformDropout(Function):
    def __init__(self,a=0.0,b=1.0,train=False, inplace=False):
        super(UniformDropout, self).__init__()
        self.a=a
        self.b=b
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
            self.noise = np.random.uniform(self.a, self.b, size=input.size())/((self.b-self.a)/2.0)
            self.noise = torch.Tensor(self.noise)
            if input.is_cuda:
                self.noise = self.noise.cuda()
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
        self.dropout_layer1 = UniformDropout(args.a, args.b, self.training, False)
        self.dropout_layer2 = UniformDropout(args.a, args.b, self.training, False)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
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

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)  #,weight_decay=1e-8

mom_epoch_interval=500.0
mom_start=0.5
mom_end=0.99

def train(epoch):
    model.train()
    model.dropout_layer1 = UniformDropout(args.a, args.b, model.training, False)
    model.dropout_layer2 = UniformDropout(args.a, args.b, model.training, False)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.998
        if epoch < 500:
            param_group['momentum'] = 0.5 * (1. - epoch / 500.) + 0.99 * (epoch / 500.)
        else:
            param_group['momentum'] =0.99
        #print('momentum:{},lr:{}'.format(param_group['momentum'],param_group['lr']))

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        #data, target = Variable(data), Variable(target)
        output = model(data)
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        if args.init_clip_max_norm is not None:
            utils.clip_grad_norm_(model.parameters(), max_norm=args.init_clip_max_norm)
        optimizer.step()

def test(epoch,error_list):
    model.eval()
    model.dropout_layer1 = UniformDropout(args.a, args.b, model.training, False)
    model.dropout_layer2 = UniformDropout(args.a, args.b, model.training, False)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            #data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('Test Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Error: {}'.format(
        epoch,test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), len(test_loader.dataset) - correct))
    print_log('{}'.format(len(test_loader.dataset) - correct),log)
    error_list.append(len(test_loader.dataset) - correct)

if __name__=='__main__':
    # mark start time
    t_start=time.time()
    save_dir='results_uniform_dropout'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    current_filename = os.path.basename(__file__)[:-3]
    log = open(os.path.join(save_dir,current_filename+'_a_'+str(args.a)+'_b_'+str(args.b)+'_lr_'+str(args.lr)+'_'+
                            time.strftime("%Y%m%d%H%M",time.localtime())+'.txt'), 'w')
    print_log('model:{}'.format(model), log)
    # 打印网络超参数
    log.write('batch_size:{}  ,epochs:{}  ,initial lr:{}  ,momentum:{}  ,log_interval:{}, optimizer:{}\n'.format(
        args.batch_size, args.epochs, optimizer.param_groups[0]['lr'], args.momentum, args.log_interval,
        str(optimizer.__class__.__name__)))
    # 训练
    # 记录每个epoch的结果
    error_list=[]
    epoch_time_start=time.time()
    min_error=300
    for epoch in range(args.epochs):  # args.epochs
        train(epoch)
        test(epoch,error_list)
        if np.min(error_list)<min_error:
            min_error=np.min(error_list)
            torch.save(model, os.path.join(save_dir, current_filename + '_a_' + str(args.a) + '_b_' + str(args.b) +
                                           '_lr_' + str(args.lr) + '_' + str(min_error) + '.pth'))
        print('cost time for epoch:{}'.format(time.time()-epoch_time_start))
        epoch_time_start = time.time()

    error_list_npy=np.array(error_list)
    error_list_npy.sort()

    print_log('cost time:{:.2f}'.format(time.time()-t_start),log)
    log.close()
