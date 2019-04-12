# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import numpy as np
from torch.autograd import Function

class Gaussian_Dropout(Function):
    def __init__(self,mean=0.5,std=0.2,train=False, inplace=False):
        super(Gaussian_Dropout, self).__init__()
        self.mean=mean
        self.std=std
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
        if self.train:
            self.noise = np.random.normal(self.mean, np.sqrt(self.std), size=input.size()).clip(0,1)/args.mu
            self.noise=torch.Tensor(self.noise)
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


