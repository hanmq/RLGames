# @Time    : 2022/12/6 15:31
# @Author  : mihan
# @File    : test.py


import numpy as np
import gym
from agent.dqn import DQN

env = gym.make('CartPole-v1', render_mode="human")

state, _ = env.reset(seed=42)

config = {
    'state_dim': 4,
    'action_dim': 2,
    'save_dir': '',
    'exploration_rate': 0.1,
    'exploration_rate_decay': 0.9998,
    'exploration_rate_min': 0.1,
    'burnin': 1e4,
    'learn_every': 20,
    'sync_every': 1e4,
    'save_every': 5e5,
    'gamma': 0.998
}

agent = DQN(config)
agent.load(r'C:\Users\mihan\Desktop\match_net_0731_cur983.340_max983.340.chkpt')

res = 0
while True:
    action = agent.select_action(state)
    state, reward, done, _, _ = env.step(action)

    res += reward
    if done:

        break

print(res)





