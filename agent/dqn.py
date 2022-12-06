# @Time    : 2022/12/5 20:39
# @Author  : mihan
# @File    : dqn.py
# @Email   : mihan@lexin.com

from pathlib import Path

import numpy as np
import torch

from replay.replay_buffer import ReplayBuffer
from network.mlp import MLP


class DQN(object):
    """
    处理离散
    """
    def __init__(self, config):
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']

        # 日志保存地址
        self.save_dir = config['save_dir']

        self.exploration_rate = config['exploration_rate']
        self.exploration_rate_decay = config['exploration_rate_decay']
        self.exploration_rate_min = config['exploration_rate_min']
        self.curr_step = 0

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.net = MLP(self.state_dim, self.action_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.MSELoss()

        # 更新相关配置
        self.burnin = config['burnin']  # min. experiences before training
        self.learn_every = config['learn_every']  # no. of experiences between updates to Q_online
        self.sync_every = config['sync_every']  # no. of experiences between Q_target & Q_online sync
        self.save_every = config['save_every']  # no. of experiences between saving Net

        buffer_capacity = 100000 if 'buffer_capacity' not in config.keys() else config['capacity']
        buffer_size = 1024 if 'buffer_size' not in config.keys() else config['buffer_size']
        self.memory = ReplayBuffer(self.state_dim, device=self.device, capacity=buffer_capacity, batch_size=buffer_size)

        self.gamma = config['gamma']

        # 平均 reward 记录
        self.curr_reward = 0
        self.max_reward = 0

    def update(self):
        if self.curr_step < self.burnin:
            return None, None

        # 把 online 的参数同步到 target 上
        if self.curr_step % self.sync_every == 0:
            self.net.sync_q_target()

        # 保持模型文件
        if self.curr_step % self.save_every == 0:
            self.save()

        # 每隔 learn_every 步更新一次模型，
        if self.curr_step % self.learn_every != 0:
            return None, None

        est_lst = []
        loss_lst = []

        state, action, reward, next_state, mask = self.memory.sample()
        td_est = self.td_estimate(state, action)
        td_trg = self.td_target(reward, next_state, mask)

        loss = self.update_q_online(td_est, td_trg)
        est_lst.append(td_est.mean().item())
        loss_lst.append(loss)

        return np.mean(est_lst), np.mean(loss_lst)

    def predict(self, state):
        b, n = state.shape[0], state.shape[1]

        # 各资方的 Q_value
        channel_values = self.net(state.view(b * n, -1), model="online").view(b, n)
        action = channel_values.argsort(dim=1, descending=True)

        return action

    def select_action(self, state):
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            # 随机返回资方排序
            action = self.random_action()
        # EXPLOIT
        else:
            action = self.max_action(state)

        if self.curr_step >= self.burnin:
            # 衰减 exploration 的概率，最低为 exploration_rate_min
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)

        # increment step
        self.curr_step += 1

        return action

    def random_action(self):
        action = np.random.randint(self.action_dim)
        return action

    def max_action(self, state):
        action = self.net(torch.tensor(state, device=self.device, dtype=torch.float), model="online").argmax().cpu().item()
        return action

    def td_estimate(self, state, action):
        # q(s, item)
        current_item_q = self.net(state, model="online").gather(1, action.view(-1, 1))
        return current_item_q

    @torch.no_grad()
    def td_target(self, reward, next_state, mask):
        target_q = self.net(next_state, model='target').max(dim=1, keepdim=True)[0]
        target = reward.view(-1, 1) + mask.view(-1, 1) * target_q
        return target

    def update_q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        for param in self.net.online.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        return loss.item()

    def update_reward(self, mean_ep_reward):
        self.curr_reward = mean_ep_reward
        if self.curr_reward > self.max_reward:
            self.max_reward = self.curr_reward

    def save(self):
        path = self.save_dir / 'model_file'
        path.mkdir(parents=True, exist_ok=True)

        file_num = int(self.curr_step // self.save_every)
        path = f'{path}/match_net_{file_num:04}_cur{self.curr_reward:.3f}_max{self.max_reward:.3f}.chkpt'
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            path
        )

        print(f"MatchNet saved to {self.save_dir} at step {self.curr_step}")

    def load(self, path):
        model = torch.load(path, map_location=self.device)
        self.net.load_state_dict(model['model'])
