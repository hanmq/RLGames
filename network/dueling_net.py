# @Time    : 2022/12/17 17:24
# @Author  : mihan
# @File    : dueling_net.py


import math

from typing import Union, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.noisy_net import NoisyLinear
from network.cnn import CNN


class DuelingNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, noisy=True):
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
        self.noisy = noisy

        lc_f = NoisyLinear if noisy else nn.Linear

        for i, o in zip(in_dims[:-2], out_dims[:-2]):
            self.feather_layer_lst.append(lc_f(i, o))

        # 最后两层网络，advantage 和 value 分不同参数，之前两个共享参数
        self.advantage_layer2 = lc_f(in_dims[-2], out_dims[-2])
        self.value_layer2 = lc_f(in_dims[-2], out_dims[-2])

        self.advantage_layer1 = lc_f(in_dims[-1], out_dims[-1])
        self.value_layer1 = lc_f(in_dims[-1], 1)

    def forward(self, x):
        for lc in self.feather_layer_lst:
            x = F.relu(lc(x))

        advantage = self.advantage_layer1(F.relu(self.advantage_layer2(x)))
        value = self.value_layer1(F.relu(self.value_layer2(x)))

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

    def reset_noise(self):
        """Reset all noisy layers."""
        if self.noisy:
            for lc in self.feather_layer_lst:
                lc.reset_noise()

            self.advantage_layer2.reset_noise()
            self.value_layer2.reset_noise()

            self.advantage_layer1.reset_noise()
            self.value_layer1.reset_noise()


input_type = Union[List, Tuple]


class DuelingCNN(nn.Module):
    def __init__(self, input_dim: input_type, output_dim, hidden_dims=None, noisy=True):
        """

        :param input_dim: (c, h, w)
        :param output_dim:
        :param hidden_dims:
        :param noisy:
        """
        super(DuelingCNN, self).__init__()
        c, h, w = input_dim
        self.noisy = noisy

        self.cnn = CNN(in_channel=c)

        input_dim = self._get_cnn_out(input_dim)

        self.dueling_fc = DuelingNet(input_dim, output_dim, hidden_dims, noisy)

    def forward(self, x):
        x = self.cnn(x)
        x = self.dueling_fc(x)
        return x

    def reset_noise(self):
        if self.noisy:
            self.dueling_fc.reset_noise()

    def _get_cnn_out(self, input_dim):
        o = self.cnn(torch.ones(1, *input_dim))
        return int(np.prod(o.size()))
