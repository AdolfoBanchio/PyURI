# Models extrated from 
# https://github.com/schneimo/ddpg-pytorch/blob/master/utils/nets.py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)

class Actor(nn.Module):
    def __init__(self, state_dim: int, 
                 action_dim: int, 
                 sizes: list,
                 max_action):
        super().__init__()
        
        self.l1 = nn.Sequential(
            nn.Linear(state_dim, sizes[0]), 
            nn.LayerNorm(sizes[0]),
            nn.LeakyReLU()
            )
        
        self.l2 = nn.Sequential(
            nn.Linear(sizes[0], sizes[1]),
            nn.LayerNorm(sizes[1]), 
            nn.LeakyReLU(),
            )
        
        self.out = nn.Linear(sizes[1], action_dim)

        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.out.bias,   -3e-4, 3e-4)


    def forward(self, s):
        
        x = s

        # layer 1
        x = self.l1(x)

        # layer 2
        x = self.l2(x)

        return torch.tanh(self.out(x))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, sizes: list = [64, 64]):
        super(Critic, self).__init__()
        num_outputs = action_dim

        # Layer 1
        self.linear1 = nn.Linear(state_dim, sizes[0])
        self.ln1 = nn.LayerNorm(sizes[0])

        # Layer 2
        # In the second layer the actions will be inserted also 
        self.linear2 = nn.Linear(sizes[0] + num_outputs, sizes[1])
        self.ln2 = nn.LayerNorm(sizes[1])

        # Output layer (single value)
        self.V = nn.Linear(sizes[1], 1)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.V.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.V.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.leaky_relu(x)

        # Layer 2
        x = torch.cat((x, actions), 1)  # Insert the actions
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.leaky_relu(x)

        # Output
        V = self.V(x)
        return V
