# @Time    : 2022/12/5 20:39
# @Author  : mihan
# @File    : dqn.py
# @Email   : mihan@lexin.com

from pathlib import Path

import torch

from replay.replay_buffer import ReplayBuffer
from network.mlp import MLP

class DQN(object):

    def __init__(self, state_dim, action_dim, save_dir: Path):
        self.state_dim = state_dim
        self.save_dir = save_dir

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.999999875
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.net = MLP(state_dim, action_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()


