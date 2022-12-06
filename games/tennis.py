# @Time    : 2022/7/20 17:12
# @Author  : mihan
# @File    : Tennis.py
# @Email   : mihan@lexin.com

import gym
from gym.utils.play import play


# CartPole-v1
# env = gym.make('CartPole-v1')
# env.action_space.seed(42)
#
# print(env.observation_space)

import gym

# env = gym.make("LunarLander-v2", render_mode="human", new_step_api=True)
import tianshou

print(gym.__version__)

# env = gym.make('CartPole-v1', render_mode="human", new_step_api=True)
# env.action_space.seed(42)
#
# observation, info = env.reset(seed=42, return_info=True)
#
# for i in range(1000):
#     # print()
#     observation, reward, done, info, _ = env.step(env.action_space.sample())
#
#     if done:
#         observation, info = env.reset(return_info=True)
#
#     print(i, observation, reward, done, info)
# env.close()
#
#
