# @Time    : 2022/12/29 15:10
# @Author  : mihan
# @File    : cnn.py

import math

import numpy as np
import cv2

import torch
import torch.nn as nn

import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channel=4):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)
