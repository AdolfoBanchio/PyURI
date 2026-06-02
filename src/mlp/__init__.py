"""
Multi-layer perceptron models for actors and critics.
"""

from .MLP_models import ValueCriticInvPen, TwinCritic, TwinCriticInvPen

__all__ = ["ValueCriticInvPen",
           "TwinCritic",
           "TwinCriticInvPen",
           ]

