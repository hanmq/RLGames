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

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.online = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),

            nn.Linear(64, 16),
            nn.LayerNorm(16),
            nn.ReLU(),

            nn.Linear(16, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        # target 全程关闭 bn 和 grad
        self.target.eval()

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, model='online'):
        if model == "online":
            return self.online(x)
        elif model == "target":
            return self.target(x)

    def sync_q_target(self):
        self.target.load_state_dict(self.online.state_dict())
