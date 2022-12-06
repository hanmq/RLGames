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
from utils.logger import MetricLogger


def train(env, agent, logger, episodes=1000):
    for e in range(episodes):
        state, _ = env.reset(seed=42)

        while True:
            action = agent.select_action(state)

            next_state, reward, done, _, _ = env.step(action)

            agent.memory.push(state, action, reward, next_state, 0 if done else agent.gamma)

            q, loss = agent.update()

            logger.log_step(reward, loss, q)

            state = next_state

            if agent.curr_step % 500 == 0:
                mean_ep_reward = logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)

                # 5w step 之后，开始记录平均 reward
                if agent.curr_step > 50000:
                    agent.update_reward(mean_ep_reward)

            if done:
                break
        logger.log_episode()


def main():
    save_dir = Path("./log") / datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make('CartPole-v1')
    env.action_space.seed(42)

    config = {
        'state_dim': 4,
        'action_dim': 2,
        'save_dir': save_dir,
        'exploration_rate': 1,
        'exploration_rate_decay': 0.9998,
        'exploration_rate_min': 0.1,
        'burnin': 1e4,
        'learn_every': 20,
        'sync_every': 1e4,
        'save_every': 5e5,
        'gamma': 0.998
    }

    agent = DQN(config)
    logger = MetricLogger(save_dir, config)

    train(env, agent, logger)


if __name__ == '__main__':
    main()
