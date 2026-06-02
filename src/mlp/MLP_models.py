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

class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Q1 Architecture
        self.l1_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1_1 = nn.LayerNorm(hidden_dim)
        self.l2_1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2_1 = nn.LayerNorm(hidden_dim)
        self.l3_1 = nn.Linear(hidden_dim, 1)

        # Q2 Architecture
        self.l1_2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1_2 = nn.LayerNorm(hidden_dim)
        self.l2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2_2 = nn.LayerNorm(hidden_dim)
        self.l3_2 = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Inicialización ortogonal para RL es excelente
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        # Asegurar que los inputs estén en el mismo device que el Critic
        device = self.l1_1.weight.device
        state = state.to(device)
        action = action.to(device)

        is_3d = state.dim() == 3
        if is_3d:
            B, T, D_s = state.shape
            _, _, D_a = action.shape
            state = state.reshape(-1, D_s)
            action = action.reshape(-1, D_a)

        # Q1 Forward
        x1 = torch.cat([state, action], dim=-1)
        x1 = F.relu(self.ln1_1(self.l1_1(x1)))
        x1 = F.relu(self.ln2_1(self.l2_1(x1)))
        q1 = self.l3_1(x1)

        # Q2 Forward
        x2 = torch.cat([state, action], dim=-1)
        x2 = F.relu(self.ln1_2(self.l1_2(x2)))
        x2 = F.relu(self.ln2_2(self.l2_2(x2)))
        q2 = self.l3_2(x2)

        if is_3d:
            q1 = q1.view(B, T, 1)
            q2 = q2.view(B, T, 1)

        return q1, q2

    def q1_forward(self, state, action):
        """Versión optimizada para el update del Actor"""
        device = self.l1_1.weight.device
        state = state.to(device)
        action = action.to(device)

        is_3d = state.dim() == 3
        if is_3d:
            B, T, D_s = state.shape
            D_a = action.shape[-1]
            state = state.reshape(-1, D_s)
            action = action.reshape(-1, D_a)

        x1 = torch.cat([state, action], dim=-1)
        x1 = F.relu(self.ln1_1(self.l1_1(x1)))
        x1 = F.relu(self.ln2_1(self.l2_1(x1)))
        q1 = self.l3_1(x1)

        if is_3d:
            q1 = q1.view(B, T, 1)
        return q1


class TwinCriticInvPen(nn.Module):
    """
    Twin Q-critic for InvertedPendulum-v5 (state_dim=4, action_dim=1).
    Same architecture pattern as TwinCritic: two independent Q networks with
    LayerNorm + ReLU hidden layers, orthogonal init. Supports 2-D (B, D)
    and 3-D (B, T, D) inputs as required by TD3+BPTT updates.
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()

        # Q1 Architecture
        self.l1_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1_1 = nn.LayerNorm(hidden_dim)
        self.l2_1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2_1 = nn.LayerNorm(hidden_dim)
        self.l3_1 = nn.Linear(hidden_dim, 1)

        # Q2 Architecture
        self.l1_2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1_2 = nn.LayerNorm(hidden_dim)
        self.l2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2_2 = nn.LayerNorm(hidden_dim)
        self.l3_2 = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        device = self.l1_1.weight.device
        state = state.to(device)
        action = action.to(device)

        is_3d = state.dim() == 3
        if is_3d:
            B, T, D_s = state.shape
            _, _, D_a = action.shape
            state = state.reshape(-1, D_s)
            action = action.reshape(-1, D_a)

        x1 = torch.cat([state, action], dim=-1)
        x1 = F.relu(self.ln1_1(self.l1_1(x1)))
        x1 = F.relu(self.ln2_1(self.l2_1(x1)))
        q1 = self.l3_1(x1)

        x2 = torch.cat([state, action], dim=-1)
        x2 = F.relu(self.ln1_2(self.l1_2(x2)))
        x2 = F.relu(self.ln2_2(self.l2_2(x2)))
        q2 = self.l3_2(x2)

        if is_3d:
            q1 = q1.view(B, T, 1)
            q2 = q2.view(B, T, 1)

        return q1, q2

    def q1_forward(self, state, action):
        device = self.l1_1.weight.device
        state = state.to(device)
        action = action.to(device)

        is_3d = state.dim() == 3
        if is_3d:
            B, T, D_s = state.shape
            D_a = action.shape[-1]
            state = state.reshape(-1, D_s)
            action = action.reshape(-1, D_a)

        x1 = torch.cat([state, action], dim=-1)
        x1 = F.relu(self.ln1_1(self.l1_1(x1)))
        x1 = F.relu(self.ln2_1(self.l2_1(x1)))
        q1 = self.l3_1(x1)

        if is_3d:
            q1 = q1.view(B, T, 1)
        return q1


class ValueCriticInvPen(nn.Module):
    """
    State-value critic V(s) for InvertedPendulum-v5 (state_dim=4).
 
    Two hidden layers with LayerNorm + ReLU and orthogonal init —
    same conventions as TwinCriticInvPen, but outputs a scalar V(s)
    instead of Q(s, a).  Supports 2-D (B, D) and 3-D (B, T, D) input
    so it can be called on flat and sequence batches without reshaping
    at the call site.
 
    Args:
        state_dim:  Observation dimensionality (default 4 for InvPen-v5).
        hidden_dim: Width of each hidden layer.
    """
 
    def __init__(self, state_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.l1  = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.l2  = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.l3  = nn.Linear(hidden_dim, 1)
        self.apply(self._init_weights)
 
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
 
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim) or (B, T, state_dim).
 
        Returns:
            Value tensor shaped (B, 1) or (B, T, 1).
        """
        is_3d = state.dim() == 3
        if is_3d:
            B, T, D = state.shape
            state = state.reshape(-1, D)
 
        x = F.relu(self.ln1(self.l1(state)))
        x = F.relu(self.ln2(self.l2(x)))
        v = self.l3(x)
 
        if is_3d:
            v = v.view(B, T, 1)
        return v