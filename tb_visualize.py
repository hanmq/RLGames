# @Time    : 2022/12/30 16:35
# @Author  : mihan
# @File    : tb_visualize.py

import torch
from torch.utils.tensorboard import SummaryWriter

from network.dueling_net import DuelingCNN


input_dim = (1, 80, 80)
output_dim = 6

net = DuelingCNN(input_dim, output_dim, noisy=False)


input_tensor = torch.ones(1, *input_dim)
writer = SummaryWriter(log_dir='./test_dir', comment='DuelingCNN')

writer.add_graph(net, [input_tensor, ])



# C:\Users\mihan\Anaconda3\python.exe C:\Users\mihan\Anaconda3\Lib\site-packages\tensorboard\main.py --logdir "D:\PycharmProjects\RLGames\runs"


