# @Time    : 2022/7/21 20:04
# @Author  : mihan
# @File    : mlp.py
# @Email   : mihan@lexin.com

import copy

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=None):
        super().__init__()

        if hidden_dims == None:
            hidden_dims = [64, 32, 16]

        self.linear_lst = nn.ModuleList()

        in_dims = [input_dim] + hidden_dims[:-1]
        for i, o in zip(in_dims, hidden_dims):
            self.linear_lst.append(
                nn.Sequential(
                    nn.Linear(i, o),
                    nn.LayerNorm(o),
                    nn.ReLU()
                )
            )

        self.last_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        for l in self.linear_lst:
            x = l(x)
        return self.last_layer(x)
