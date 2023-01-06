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


class DuelCNN(nn.Module):
    """
    CNN with Duel Algo. https://arxiv.org/abs/1511.06581
    """

    def __init__(self, h, w, output_size):
        super(DuelCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        convw, convh = self.conv2d_size_calc(w, h, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=3, stride=1)

        linear_input_size = convw * convh * 64  # Last conv layer's out sizes

        # Action layer
        self.Alinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Alrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Alinear2 = nn.Linear(in_features=128, out_features=output_size)

        # State Value layer
        self.Vlinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Vlrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Vlinear2 = nn.Linear(in_features=128, out_features=1)  # Only 1 node

    def conv2d_size_calc(self, w, h, kernel_size=5, stride=2):
        """
        Calcs conv layers output image sizes
        """
        next_w = (w - (kernel_size - 1) - 1) // stride + 1
        next_h = (h - (kernel_size - 1) - 1) // stride + 1
        return next_w, next_h

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten every batch

        Ax = self.Alrelu(self.Alinear1(x))
        Ax = self.Alinear2(Ax)  # No activation on last layer

        Vx = self.Vlrelu(self.Vlinear1(x))
        Vx = self.Vlinear2(Vx)  # No activation on last layer

        q = Vx + (Ax - Ax.mean())

        return q
