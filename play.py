# @Time    : 2022/12/6 15:31
# @Author  : mihan
# @File    : test.py


import numpy as np
import gym
from agent.dqn import DQN
from agent.noisy_dqn import NoisyDQN
from gym.utils.play import play

env = gym.make('LunarLander-v2', render_mode="human")

state, _ = env.reset()

config = {
    'state_dim': 8,
    'action_dim': 4,
    'save_dir': '',
    'exploration_rate': 0.1,
    'exploration_rate_decay': 0.9998,
    'exploration_rate_min': 0.1,
    'burnin': 1e4,
    'learn_every': 20,
    'sync_every': 1e4,
    'save_every': 5e5,
    'gamma': 0.998,
    'update_mode': 'hard',
    'lr': 0.001,
    'hidden_dims': [128, 128],
    # 'loss_fn': torch.nn.SmoothL1Loss(),
    'netword': 'local'  # 自己实现的网络结构
}

agent = NoisyDQN(config)
agent.load(r'model_file/noisy_dqn.chkpt')

res = 0
while True:
    action = agent.max_action(state)
    state, reward, done, _, _ = env.step(action)

    res += reward
    print(reward)
    if done:

        break

print(res)
