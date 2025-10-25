import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PPOCritic(nn.Module):
    """Value function estimator for PPO"""
    def __init__(self, state_dim: int, hidden_sizes: list = [64, 64]):
        super().__init__()
        
        if len(hidden_sizes) < 2:
            raise ValueError("Critic expects at least two hidden layer sizes")
            
        layers = []
        # Input normalization
        layers.append(nn.LayerNorm(state_dim))
        
        # First hidden layer
        layers.append(nn.Linear(state_dim, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Additional hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, np.sqrt(2))
            module.bias.data.zero_()
            
    def forward(self, state):
        return self.net(state)