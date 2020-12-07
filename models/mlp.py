# Source: https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/cifar10_models/resnet.py
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=10, hidden_units=[100, 100, 100], activation='tanh', bias=True):
        super(MLP, self).__init__()
        assert activation in ['tanh', 'relu']
        if activation == 'tanh':
            self.activation_function = torch.tanh
        if activation == 'relu':
            self.activation_function = torch.relu

        self.num_classes = num_classes
        self.input_dim = input_dim
        last_dim = input_dim
        self.fcs = []
        for i, n_h in enumerate(hidden_units):
            self.fcs.append(nn.Linear(last_dim, n_h, bias=bias))
            self.add_module(f"hidden_layer_{i}", self.fcs[-1])
            last_dim = n_h
            
        self.logit_fc = nn.Linear(last_dim, self.num_classes, bias=bias)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for fc in self.fcs:
            x = fc(x)
            x = self.activation_function(x)
        x = self.logit_fc(x)
        return x
    
def get_mlp(*args, **kwargs):
    return MLP(*args, **kwargs)