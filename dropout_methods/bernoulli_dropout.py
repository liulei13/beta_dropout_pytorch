# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import numpy as np
from torch.autograd import Function


class Bernoulli_Dropout(Function):
    def __init__(self,p=0.5,train=False, inplace=False):
        super(Bernoulli_Dropout, self).__init__()
        self.p=p
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
            self.noise = self._make_noise(input)
            self.noise.bernoulli_(1 - self.p).div_(1 - self.p)
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
