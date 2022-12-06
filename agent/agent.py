from pathlib import Path

import torch
import numpy as np

from network.mlp import MLP
from replay.replay_buffer import ReplayBuffer


class Match(object):
    """
    目前实现的是 Double DQN 的训练方式
    """

    def __init__(self, state_dim, save_dir: Path, pass_rate_idx):
        self.state_dim = state_dim
        self.save_dir = save_dir

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.999999875
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        # 网络相关
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.net = MLP(state_dim)
        self.net.to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.batch_size = 64

        self.memory = ReplayBuffer(self.batch_size, self.device)

        # 计算 reward 的衰减系数， 0.9999 时，在 20000 笔订单之后，衰减为 0.1353
        self.gamma = 0.9999

        # 通过率特征的索引位置
        self.pass_rate_idx = pass_rate_idx

        # 更新相关配置
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 20  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync
        self.save_every = 5e5  # no. of experiences between saving Net

    def update(self):
        if self.curr_step < self.burnin:
            return None, None

        # 把 online 的参数同步到 target 上
        if self.curr_step % self.sync_every == 0:
            self.net.sync_q_target()

        # 保持模型文件
        if self.curr_step % self.save_every == 0:
            self.save()

        # 每隔 learn_every 步，才更新一次模型，
        if self.curr_step % self.learn_every != 0:
            return None, None

        state, reward, next_state, done = self.memory.sample()
        td_est = self.td_estimate(state)
        td_trg = self.td_target(reward, next_state, done)

        loss = self.update_q_online(td_est, td_trg)
        return td_est.mean().item(), loss

    def predict(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float)
        # 各资方的 Q_value
        channel_values = self.net(state, model="target")
        action = channel_values.argsort(dim=0, descending=True)[:, 0].cpu().numpy()
        return action

    def select_action(self, state):

        if state.shape[0] == 1:
            # 只有一个资方时，就不需要输出排序了
            action = np.array([0])

        else:
            # EXPLORE
            if np.random.rand() < self.exploration_rate:
                # 随机返回资方排序
                action = np.arange(state.shape[0])
                np.random.shuffle(action)

            # EXPLOIT
            else:
                with torch.no_grad():
                    state = torch.tensor(state, device=self.device, dtype=torch.float)
                    # 各资方的 Q_value
                    channel_values = self.net(state, model="online")
                    action = channel_values.argsort(dim=0, descending=True)[:, 0].cpu().numpy()

        # 衰减 exploration 的概率，当 <= 最小值时，就不再衰减了
        if self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate *= self.exploration_rate_decay

        # increment step
        self.curr_step += 1

        return action

    def td_estimate(self, state):
        # q(s, item)
        current_item_q = self.net(state, model="online")
        return current_item_q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        batch_size, max_channel_len, feature_size = next_state.shape

        # 这里由于 next_state padding 里大量的 0 值，开启 train 状态，会影响 BN 层状态量统计
        self.net.online.eval()
        # batch_size * max_channel_len
        next_item_values = self.net(next_state.view(batch_size * max_channel_len, -1),
                                    model="online").view(batch_size, -1)
        self.net.online.train()

        # 每一个 batch，都按照 channel 对应的 item_q_value 降序排序，得到最好的排序
        best_action = next_item_values.argsort(descending=True)

        # 由于 torch 目前没有找到按照 2d tensor 索引的方式
        # 这里把 arg_sort 和 item_values 拉成 (-1, 1) 的形状，索引完之后，再 view 回去
        best_action.add_(torch.arange(0, batch_size * max_channel_len, step=max_channel_len,
                                      device=self.device).view(-1, 1))
        best_action = best_action.view(-1)

        # Double DQN：用 online 选择一个最大价值的动作，然后用 target 算 value，避免估算的 q 过大的问题
        target_q = self.net(next_state.view(batch_size * max_channel_len, -1),
                            model="target")[best_action].view(batch_size, -1)

        # 计算匹配到各资方概率
        pass_rate = next_state[:, :, self.pass_rate_idx].view(-1, 1)[best_action].view(batch_size, -1)
        not_pass_rate = 1 - pass_rate
        prob = pass_rate * torch.cat([torch.ones((batch_size, 1), device=self.device),
                                      torch.cumprod(not_pass_rate, dim=1)[:, :-1]], dim=1)
        prob /= prob.sum(dim=1, keepdim=True)

        target = reward + self.gamma * torch.sum(target_q * prob, dim=1, keepdim=True) * (1 - done)
        return target

    def update_q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.net.online.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        return loss.item()

    def save(self):
        path = self.save_dir / 'model_file'
        path.mkdir(parents=True, exist_ok=True)

        path = f'{path}/match_net_{int(self.curr_step // self.save_every)}.chkpt'
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            path
        )

        print(f"MatchNet saved to {self.save_dir} at step {self.curr_step}")

    def load(self, path):
        self.net.load_state_dict(torch.load(path)['model'])
        self.net.to(self.device)
