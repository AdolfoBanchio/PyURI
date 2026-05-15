"""
Multi-layer perceptron models for actors and critics.
"""

from .MLP_models import Actor, Critic, BestCritic, TwinCritic, TwinCriticInvPen

__all__ = ["Actor",
           "Critic",
           "BestCritic",
           "TwinCritic",
           "TwinCriticInvPen",
           ]
