# @Time    : 2022/12/6 19:38
# @Author  : mihan
# @File    : trainer.py

import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datetime import datetime
import numpy_financial as npf
import pandas as pd
import torch

from pathlib import Path

import gym

from agent.dqn import DQN
from agent.noisy_dqn import NoisyDQN
from agent.ddqn import DoubleDQN
from utils.logger import MetricLogger


def train(env, agent, logger, episodes=1000):
    for e in range(episodes):
        state, _ = env.reset()

        i = 0
        while True:
            if i >= 1000:
                break
            i += 1
            action = agent.select_action(state)

            next_state, reward, done, _, _ = env.step(action)

            # 减少卡住，一直浮空的情况
            if i == 1000:
                reward = -100

            agent.memory.push(state, action, reward, next_state, done)

            q, loss = agent.update()

            logger.log_step(reward, loss, q)

            state = next_state

            if agent.curr_step % 500 == 0:
                mean_ep_reward = logger.record(episode=e, epsilon=0, step=agent.curr_step)

                # 5w step 之后，开始记录平均 reward
                if agent.curr_step > 50000:
                    agent.update_reward(mean_ep_reward)

            if done:
                break

        # agent.reset_noise()
        logger.log_episode()


def main():
    env_name = 'LunarLander-v2'

    save_dir = Path("./log") / (datetime.now().strftime("%Y_%m_%d_%H:%M:%S") + env_name + 'noisy_dqn')
    save_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_name)

    config = {
        'state_dim': env.observation_space.shape[0],
        'action_dim': env.action_space.n,
        'save_dir': save_dir,
        'burnin': 256,
        'learn_every': 4,
        'sync_every': 150,
        'save_every': 5000,
        'gamma': 0.99,
        'buffer_capacity': 100000,
        'buffer_size': 512,
        'update_mode': 'hard',
        'lr': 0.001,
        'hidden_dims': [128, 128],
        # 相比之下，用 MSE 学习会更慢
        'loss_fn': torch.nn.SmoothL1Loss(),
        'netword': 'local'  # 自己实现的网络结构
    }

    # agent = NoisyDQN(config)
    agent = DoubleDQN(config)
    logger = MetricLogger(save_dir, config)

    train(env, agent, logger, 1000)


if __name__ == '__main__':
    main()
