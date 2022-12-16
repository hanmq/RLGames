## RLGames

### agent 实现历程

以下都是按照日期排序，

| 日期 | agent        | env            | trainer              | reward |
| ---- | ------------ | -------------- | -------------------- | ------ |
|      | dqn.py       | CartPole-v1    | 未保留               | inf    |
|      | noisy_dqn.py | LunarLander-v2 | noisy_dqn_trainer.py | 270    |
|      | ddqn.py      |                |                      |        |



### 调参日记（RL 调参真是太离谱了）

- 训练的时候， env.reset(seed=42) 环境固定 seed 时，好像会导致训练不稳定
- (12/11) **Epsilon-Greedy 在 LunarLander-v2 环境下，每 episode 降低一次探索率比每 step 降低一次会更稳定**：最近几天训练 DQN ，LunarLander-v2 环境，但是效果一直没有 [别人的实现](https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html) 训练稳定，发现是探索率的问题， 
  人家是每一个 episode 降低一次探索率，但我设置的是每个 step 降低一次；经测试，在 LunarLander-v2 环境下，一个 episode 下使用同一探索率训练更加稳定；
- (12/16) **noisy_net 是每 update 一次 Q net 之后，就重新采样一次噪音，不是每一个 episode 降低一次
  ，李宏毅强化学习课程讲的是每回合更新一次，是不对的，[noisyNet论文](https://openreview.net/pdf?id=rywHCPkAW) 里面也提到过这一点（3.1 小节第二段），
  我在 LunarLander-v2 环境下验证每 episode 更新一次，基本不 work ，训练过程及不稳定，
  update 之间噪音保持不变**；

