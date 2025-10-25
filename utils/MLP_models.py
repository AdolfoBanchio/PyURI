import math
import torch
import torch.nn as nn
import torch.nn.functional as F

""" 
Models extracted from
https://arxiv.org/pdf/1509.02971
paper that introduces DDPG
"""
def fanin_init(tensor, fanin=None):
    if fanin is None:
        fanin = tensor.size(0)  # number of input units to this layer
    bound = 1. / math.sqrt(fanin)
    with torch.no_grad():
        nn.init.uniform_(tensor, -bound, bound)

FINAL_W_INIT = 3e-3

class Actor(nn.Module):
    """
    400-300 MLP (ReLU), tanh output rescaled to [-max_action, max_action].
    Matches common DDPG baselines.
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float, size: list[int] = [400,300]):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, size[0])
        self.fc2 = nn.Linear(size[0], size[1])
        self.out = nn.Linear(size[1], action_dim)
        self.max_action = float(max_action)

        # Init: fan-in for hidden layers, small uniform for output
        fanin_init(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        fanin_init(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.out.weight, -FINAL_W_INIT, FINAL_W_INIT)
        nn.init.uniform_(self.out.bias,   -FINAL_W_INIT, FINAL_W_INIT)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        a = torch.tanh(self.out(x))                 # [-1, 1]
        return a * self.max_action                   # rescale to env bounds


class Critic(nn.Module):
    """
    400-300 MLP (ReLU). Action is injected at the second layer:
      x1 = ReLU(W1 s + b1)
      x2 = ReLU(W2 [x1, a] + b2)
      Q  = W3 x2 + b3  (scalar)
    """
    def __init__(self, state_dim: int, action_dim: int, size: list[int] = [400,300]):
        super().__init__()
        self.fcs1 = nn.Linear(state_dim, size[0])               # state -> 400
        self.fcs2 = nn.Linear(size[0] + action_dim, size[1])        # [x1, a] -> 300
        self.out  = nn.Linear(size[1], 1)

        # Init
        fanin_init(self.fcs1.weight); nn.init.zeros_(self.fcs1.bias)
        fanin_init(self.fcs2.weight); nn.init.zeros_(self.fcs2.bias)
        nn.init.uniform_(self.out.weight, -FINAL_W_INIT, FINAL_W_INIT)
        nn.init.uniform_(self.out.bias,   -FINAL_W_INIT, FINAL_W_INIT)

    def forward(self, s, a):
        x = F.relu(self.fcs1(s))
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fcs2(x))
        q = self.out(x)                                   # (B, 1)
        return q
