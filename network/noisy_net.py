# @Time    : 2022/12/12 19:28
# @Author  : mihan
# @File    : noisy_net.py

import math

import torch
import torch.nn as nn

import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor, is_train=True) -> torch.Tensor:
        """
        eval 的时候，去掉噪音
        """

        if is_train:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


class NoisyNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=None, noisy_num=3):
        super().__init__()

        if hidden_dims == None:
            hidden_dims = [64, 32, 16]

        self.linear_lst = nn.ModuleList()

        in_dims = [input_dim] + hidden_dims
        out_dims = hidden_dims + [output_dim]

        # 普通全连接层
        for i, o in zip(in_dims[:-noisy_num], out_dims[:-noisy_num]):
            self.linear_lst.append(
                nn.Sequential(
                    nn.Linear(i, o),
                    nn.ReLU()
                )
            )

        self.noisy_linear_lst = nn.ModuleList()

        # noisy net 全连接层
        for i, o in zip(in_dims[-noisy_num:-1], out_dims[-noisy_num:-1]):
            self.noisy_linear_lst.append(NoisyLinear(i, o))

        self.last_layer = NoisyLinear(in_dims[-1], out_dims[-1])

    def forward(self, x, is_train=True):
        for l in self.linear_lst:
            x = l(x)

        for l in self.noisy_linear_lst:
            x = l(x, is_train=is_train)
            x = F.relu(x)

        return self.last_layer(x, is_train=is_train)

    def reset_noise(self):
        """Reset all noisy layers."""
        for l in self.noisy_linear_lst:
            l.reset_noise()

        self.last_layer.reset_noise()
