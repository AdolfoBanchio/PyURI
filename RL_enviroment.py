import gymnasium as gym
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from FIURI_node import FIURI_node
from bindsnet.network import network


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

def set_seed(env, seed: int = 42):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


env = gym.make("MountainCarContinuous-v0", 
               render_mode="rgb_array")


Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object) :
    def __init__(self, capacity) :
        self.memory = deque([], maxlen=capacity)
    def push(self, *args) :
        self.memory.append(Transition(*args))
    def sample(self, batch_size) :
        return random.sample(self.memory, batch_size)
    def __len__(self) :
        return len(self.memory)
    

def select_action(state) :
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold :
        with torch.no_grad() :
            return policy_net(state).max(1)[1].view(1, 1), eps_threshold
    else :
        ## Random Action
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long), eps_threshold
