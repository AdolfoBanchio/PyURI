# Models extrated from 
# https://github.com/schneimo/ddpg-pytorch/blob/master/utils/nets.py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim: int, 
                 action_dim: int, 
                 sizes: list,
                 max_action):
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
        self.max_action = max_action

    def forward(self, s):
        # scale tanh output to env action range (MountainCarContinuous is [-1, 1])
        return self.net(s) * self.max_action


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, sizes: list = [64, 64]):
        super().__init__()

        if len(sizes) < 2:
            raise ValueError("Critic expects at least two hidden layer sizes.")

        self.fc1 = nn.Linear(state_dim, sizes[0])
        self.fc2 = nn.Linear(sizes[0] + action_dim, sizes[1])
        self.fc_out = nn.Linear(sizes[1], 1)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.fc1, self.fc2, self.fc_out):
            fan_in = layer.weight.size(1)
            bound = 1.0 / np.sqrt(fan_in)
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.uniform_(layer.bias, -bound, bound)

    def forward(self, s, a):
        x = F.relu(self.fc1(s))
        x = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
