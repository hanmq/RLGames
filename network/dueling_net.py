# @Time    : 2022/12/17 17:24
# @Author  : mihan
# @File    : dueling_net.py


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.noisy_net import NoisyLinear


class DuelingNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None):
        """
        最
        :param input_dim:
        :param output_dim:
        :param hidden_dims:
        """
        super().__init__()

        if hidden_dims == None:
            hidden_dims = [128, 128]

        in_dims = [input_dim] + hidden_dims
        out_dims = hidden_dims + [output_dim]

        self.feather_layer_lst = nn.ModuleList()

        for i, o in zip(in_dims[:-2], out_dims[:-2]):
            self.feather_layer_lst.append(NoisyLinear(i, o))

        # 最后两层网络，advantage 和 value 分不同参数，之前两个共享参数
        self.advantage_layer2 = NoisyLinear(in_dims[-2], out_dims[-2])
        self.value_layer2 = NoisyLinear(in_dims[-2], out_dims[-2])

        self.advantage_layer1 = NoisyLinear(in_dims[-1], out_dims[-1])
        self.value_layer1 = NoisyLinear(in_dims[-1], 1)

    def forward(self, x):
        for lc in self.feather_layer_lst:
            x = F.relu(lc(x))

        advantage = self.advantage_layer1(F.relu(self.advantage_layer2(x)))
        value = self.value_layer1(F.relu(self.value_layer2(x)))

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

    def reset_noise(self):
        """Reset all noisy layers."""
        for lc in self.feather_layer_lst:
            lc.reset_noise()

        self.advantage_layer2.reset_noise()
        self.value_layer2.reset_noise()

        self.advantage_layer1.reset_noise()
        self.value_layer1.reset_noise()
