# @Time    : 2022/7/21 18:02
# @Author  : mihan
# @File    : base.py
# @Email   : mihan@lexin.com


from collections import deque

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import random

from typing import List, Union, Tuple

SizeType = Union[Tuple, List, int]


class ReplayBuffer:

    def __init__(self, feature_size: SizeType, device=torch.device('cuda:0'), capacity=100000, batch_size=1024):

        self.capacity = capacity

        if isinstance(feature_size, int):
            size = (capacity, feature_size)
        else:
            size = (capacity, *feature_size)

        self.state = torch.empty(size=size, dtype=torch.float, device=device)
        self.action = torch.empty(size=(capacity,), dtype=torch.int64, device=device)
        self.reward = torch.empty(size=(capacity,), dtype=torch.float, device=device)

        self.next_state = torch.empty(size=size, dtype=torch.float, device=device)

        # mask = gamma if not done else 0
        self.mask = torch.empty(size=(capacity,), dtype=torch.float, device=device)

        self.device = device
        self.position = 0
        # 用于记录当前 push 了多少数据
        self.n = 0

        # 控制 push 出来数据的大小
        self.batch_size = batch_size

        # memory 是否已装满
        self.memory_full = False

    def push(self, state, action, reward, next_state, mask):

        self.state[self.position] = torch.tensor(state, device=self.device, dtype=torch.float)
        self.action[self.position] = action
        self.next_state[self.position] = torch.tensor(next_state, device=self.device, dtype=torch.float)

        self.reward[self.position] = reward
        self.mask[self.position] = mask

        self.position += 1
        if self.position >= self.capacity:
            self.position = 0
            self.memory_full = True

        self.n += 1

    def sample(self):
        max_len = self.capacity if self.memory_full else self.position
        idx = np.random.randint(0, max_len, size=(self.batch_size,))

        state = self.state[idx]
        action = self.action[idx]
        reward = self.reward[idx]
        next_state = self.next_state[idx]
        mask = self.mask[idx]

        return state, action, reward, next_state, mask


