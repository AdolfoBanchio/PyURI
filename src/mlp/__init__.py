"""
Multi-layer perceptron models for actors and critics.
"""

from .MLP_models import Actor, Critic, BestCritic

__all__ = ["Actor", 
           "Critic",
           "BestCritic"
           ]
