# Models extrated from 
# https://github.com/schneimo/ddpg-pytorch/blob/master/utils/nets.py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, sizes: list = [64, 64]):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape=state_dim),
            nn.Linear(state_dim, sizes[0]), 
            nn.ReLU(),
            nn.Linear(sizes[0], sizes[1]),
            nn.ReLU(),
            nn.Linear(sizes[1], action_dim), 
            nn.Tanh()
        )

    def forward(self, s):
        # scale tanh output to env action range (MountainCarContinuous is [-1, 1])
        return self.net(s)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, sizes: list = [64, 64]):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(state_dim + action_dim),
            nn.Linear(state_dim + action_dim, sizes[0]),
            nn.ReLU(),
            nn.Linear(sizes[0], sizes[1]),
            nn.ReLU(),
            nn.Linear(sizes[1], 1)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)  # Critical to ensure q has right shape.