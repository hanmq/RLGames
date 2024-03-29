# @Time    : 2022/12/17 18:12
# @Author  : mihan
# @File    : dueling_ddqn.py

"""
在 Double DQN 基础上加上 dueling net
"""

import copy
from pathlib import Path

import numpy as np
import torch

from replay.replay_buffer import ReplayBuffer
from network.dueling_net import DuelingNet, DuelingCNN, DuelCNN


class DuelingDQN(object):
    """
    处理离散
    """

    def __init__(self, config):
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']

        # 日志保存地址
        self.save_dir = config['save_dir']

        self.exploration_rate = config.get('exploration_rate', 1)
        self.exploration_rate_decay = config.get('exploration_rate_decay', 0.99)
        self.exploration_rate_min = config.get('exploration_rate_min', 0.02)
        self.curr_step = 0

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # True for noisy net, False for Epsilon-Greedy
        self.is_noisy = config['is_noisy']

        hidden_dims = None if 'hidden_dims' not in config.keys() else config['hidden_dims']
        self.online = DuelingCNN(
            self.state_dim, self.action_dim, hidden_dims=hidden_dims, noisy=self.is_noisy).to(self.device)

        # self.online = DuelingNet(
        #     self.state_dim, self.action_dim, hidden_dims=hidden_dims, noisy=self.is_noisy).to(self.device)

        # self.online = DuelCNN(h=80, w=80, output_size=self.action_dim).to(self.device)
        # self.online = Network(self.state_dim, self.action_dim).to(self.device)

        self.target = copy.deepcopy(self.online)
        # target 全程关闭 bn 和 grad
        self.target.eval()

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

        learning_rate = 0.001 if 'lr' not in config.keys() else config['lr']
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)

        if 'loss_fn' in config:
            self.loss_fn = config['loss_fn']
        else:
            self.loss_fn = torch.nn.SmoothL1Loss()  # torch.nn.MSELoss()

        # 更新相关配置
        self.burnin = config['burnin']  # min. experiences before training
        self.learn_every = config['learn_every']  # no. of experiences between updates to Q_online
        self.sync_every = config['sync_every']  # no. of experiences between Q_target & Q_online sync
        self.save_every = config['save_every']  # no. of experiences between saving Net

        buffer_capacity = 100000 if 'buffer_capacity' not in config.keys() else config['buffer_capacity']
        buffer_size = 1024 if 'buffer_size' not in config.keys() else config['buffer_size']
        self.memory = ReplayBuffer(self.state_dim, device=self.device, capacity=buffer_capacity, batch_size=buffer_size)

        self.gamma = config['gamma']

        # 平均 reward 记录
        self.curr_reward = 0
        self.max_reward = 0

        self.update_mode = config['update_mode']  # 目标网络的更新方式，'soft' 'hard'
        self.tau = config.get('tau', 1e-3)

    def update(self):
        if self.curr_step < self.burnin:
            return 0, 0, 0

        # 保持模型文件
        if self.curr_step % self.save_every == 0:
            self.save()

        # 每隔 learn_every 步更新一次模型，
        if self.curr_step % self.learn_every != 0:
            return 0, 0, 0

        state, action, reward, next_state, done = self.memory.sample()

        td_est = self.td_estimate(state, action)
        td_trg = self.td_target(reward, next_state, done)
        loss, total_norm = self.update_q_online(td_est, td_trg)

        q_est = td_est.mean().item()

        # state, action, reward, next_state, done = self.memory.sample()
        #
        # # Make predictions
        # state_q_values = self.online(state)
        # next_states_q_values = self.online(next_state)
        # next_states_target_q_values = self.target(next_state)
        #
        # # Find selected action's q_value
        # # selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # selected_q_value = state_q_values.gather(1, action)  # .squeeze(1)
        #
        # # Get indice of the max value of next_states_q_values
        # # Use that indice to get a q_value from next_states_target_q_values
        # # We use greedy for policy So it called off-policy
        # next_states_target_q_value = next_states_target_q_values.gather(1, next_states_q_values.max(1)[1].unsqueeze(
        #     1))  # .squeeze(1)
        # # Use Bellman function to find expected q value
        # expected_q_value = reward + self.gamma * next_states_target_q_value * (1 - done)
        #
        # # Calc loss with expected_q_value and q_value
        # loss = (selected_q_value - expected_q_value.detach()).pow(2).mean()
        #
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # q_est = selected_q_value.mean().item()
        # loss = loss.item()

        if self.update_mode == 'hard':
            # 把 online 的参数同步到 target 上
            if self.curr_step % self.sync_every == 0:
                self.sync_q_target()
        else:
            self.soft_sync(self.tau)

        # noisy net 必须是每 update 一次，更新一次 noisy ，不能是每隔
        if self.is_noisy:
            self.reset_noise()

        return q_est, loss, 0

    def select_action(self, state):

        if self.is_noisy:
            action = self.max_action(state)
        else:
            # EXPLORE
            if np.random.rand() < self.exploration_rate:
                # 随机返回资方排序
                action = self.random_action()
            # EXPLOIT
            else:
                action = self.max_action(state)

        # increment step
        self.curr_step += 1

        return action

    def update_eps(self):
        """
        在一个 episode 结束时，降低一次探索率
        :return:
        """
        # if self.curr_step >= self.burnin:
        # 衰减 exploration 的概率，最低为 exploration_rate_min
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)

    def random_action(self):
        action = np.random.randint(self.action_dim)
        return action

    def max_action(self, state):
        with torch.no_grad():
            action = self.online(torch.tensor(state, device=self.device, dtype=torch.float)).argmax().cpu().item()
        return action

    def td_estimate(self, state, action):
        # q(s, item)
        current_item_q = self.online(state).gather(1, action)
        return current_item_q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        # Double DQN：用 online 选择一个最大价值的动作，然后用 target 算 value，避免估算的 q 过大的问题
        best_action = self.online(next_state).argmax(dim=1, keepdim=True)
        target_q = self.target(next_state).gather(1, best_action)

        target = reward + self.gamma * target_q * (1 - done)
        return target

    def update_q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()

        # total_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1)

        self.optimizer.step()

        # 每次更新网络之后，都要重新采样噪音
        # self.reset_noise()
        return loss.item(), 0  # total_norm.item()

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
            dict(model=self.online.state_dict()),
            path
        )

        print(f"MatchNet saved to {self.save_dir} at step {self.curr_step}")

    def load(self, path):
        model = torch.load(path, map_location=self.device)
        self.online.load_state_dict(model['model'])

    def sync_q_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def soft_sync(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """

        for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def reset_noise(self):
        self.online.reset_noise()
        self.target.reset_noise()
