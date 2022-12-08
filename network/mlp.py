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

        in_dims = [input_dim] + hidden_dims[:1]
        out_dims = hidden_dims[1:]

        self.linear_lst = nn.ModuleList()

        for i, o in zip(in_dims, out_dims):
            self.linear_lst.append(
                nn.Sequential(
                    nn.Linear(i, o),
                    nn.LayerNorm(o),
                    nn.ReLU()
                )
            )

        self.last_layer = nn.Linear(out_dims[-1], out_dims)
        self.online = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),

            nn.Linear(16, output_dim)
        )

        self.target = copy.deepcopy(self.online)

    def forward(self, x):
        for l in self.linear_lst:
            x = l(x)
        return self.last_layer(x)
