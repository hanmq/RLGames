# @Time    : 2022/12/6 15:31
# @Author  : mihan
# @File    : test.py


import numpy as np
import gym
from agent.dqn import DQN
from agent.noisy_dqn import NoisyDQN
from agent.ddqn import DoubleDQN

from agent import DuelingDQN
import time

from gym.utils.play import play

import torch
import cv2

# env = gym.make('ALE/Pong-v5')
# state = env.reset()
# env.step(4)
#
# res = 0
# i = 0
# while True:
#
#     state, reward, done, _ = env.step(env.action_space.sample())
#     res += reward
#     i += 1
#     if done:
#
#         break
#
# print(i)
env_name = 'PongDeterministic-v4'
env = gym.make(env_name)

config = {
    'state_dim': (4, 80, 80),
    'is_noisy': False,
    'action_dim': env.action_space.n,
    'save_dir': '',
    'burnin': 40000,
    'learn_every': 1,
    'sync_every': 1e9,
    'save_every': 10000,
    'gamma': 0.97,
    'buffer_capacity': 1,
    'buffer_size': 64,
    'update_mode': 'hard',
    'lr': 0.00025,
    'hidden_dims': [128],
    'loss_fn': torch.nn.MSELoss(),
    'netword': 'local',  # 自己实现的网络结构
    'update': 'self'
}
agent = DuelingDQN(config)
agent.load(r'model_file/dueling_ddqn.chkpt')

res = 0


def process_state(state):
    # 1、转灰度  2、裁剪前后 34,16 像素  3、(160, 160) resize 成 (80, 80)   4、除以 255 归一化
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state[20:], (80, 80)) / 255
    return state


state = env.reset()
state = process_state(state)
state = np.stack((state, state, state, state))
while True:
    # env.render()
    action = agent.max_action(state[np.newaxis, :, :])

    next_state, reward, done, _ = env.step(action)
    next_state = process_state(next_state)
    next_state = np.stack((next_state, state[0], state[1], state[2]))

    res += reward
    print(reward)
    if done:
        break

    state = next_state
    # time.sleep(0.04)
print(res)


