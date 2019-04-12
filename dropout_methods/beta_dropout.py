# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import numpy as np
from torch.autograd import Function

class Beta_Dropout(Function):
    def __init__(self,A=3.0,B=3.0,train=False, inplace=False):
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
        if self.train:
            self.noise = np.random.beta(self.A, self.B, size=input.size()) / (self.A / (self.A + self.B))
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
