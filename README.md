# RLGames


调参日记（调参真是太离谱了）
- 训练的时候， env.reset(seed=42) 环境固定 seed 时，好像会导致训练不稳定
- 记一下，最近几天训练 DQN 玩LunarLander-v2，但是效果一直没有网上下载的代码训练稳定，发现是探索率的问题， 
  网上下载的代码是每一个 episode 降低一次探索率，但我代码是每隔一个 step 降低一次；经测试， 
  在 LunarLander-v2 环境下，一个 episode 下使用同一探索率更好；
  
