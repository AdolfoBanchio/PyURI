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

class BestCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Input Normalization
        # affine=False because we don't want to learn a shift, just scale.
        self.state_norm = nn.BatchNorm1d(state_dim, affine=False, track_running_stats=True)
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # Stability fix
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)  # Stability fix
        self.l3 = nn.Linear(hidden_dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        """
        Args:
            state: (Batch, State_Dim) - Raw observations [pos, vel]
            action: (Batch, Action_Dim)
        """
        is_3d = state.dim() == 3
        if is_3d:
            B, T, _ = state.shape
            state = state.reshape(-1, state.shape[-1])
            action = action.reshape(-1, action.shape[-1])

        # --- Q1 ---
        s_norm1 = self.state_norm(state)
        x1 = torch.cat([s_norm1, action], dim=1)
        x1 = F.relu(self.ln1(self.l1(x1)))
        x1 = F.relu(self.ln2(self.l2(x1)))
        q1 = self.l3(x1)
        
        # 4. Reshape Output
        if is_3d:
            q1 = q1.view(B, T, 1)
        return q1


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
