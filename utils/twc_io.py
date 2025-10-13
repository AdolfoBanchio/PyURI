# utils/twc_io_wrapper.py
import torch
import torch.nn.functional as F

# Ranges for MountainCarContinuous
POS_MIN, POS_MAX = -1.2, 0.6
VEL_MAX = 0.07   # symmetric
# Min Max states of neurons
MIN_STATE, MAX_STATE = -10,10
def _norm(x, lo, hi):
    # scale to [-1, 1]
    return 2.0 * (x - lo) / (hi - lo) - 1.0

def mcc_obs_encoder(obs: torch.Tensor, n_inputs=4, device=None):
    """
    obs: (B, 2) -> [position, velocity]
    returns ex_in, in_in of shape (B, 4): [pos+, pos-, vel+, vel-]
    """
    if device is None:
        device = obs.device
    pos = obs[:, 0]
    vel = obs[:, 1]

    pos_val = pos / POS_MAX
    vel_val = vel / VEL_MAX

    pos_n = (MAX_STATE-MIN_STATE)*pos_val + MIN_STATE
    vel_n = (MAX_STATE-MIN_STATE)*vel_val + MIN_STATE
    
    # Signed split (positive part to EX, negative magnitude to IN)
    pos_ex = F.softplus(pos_n)
    pos_in = F.softplus(-pos_n)
    vel_ex = F.softplus(vel_n)
    vel_in = F.softplus(-vel_n)
    zero = torch.zeros(pos_ex.shape)

    ex_in = torch.stack([zero , pos_ex, zero, vel_ex], dim=1)
    in_in = torch.stack([vel_in, zero, pos_in, zero], dim=1)

    return ex_in.to(device), in_in.to(device)

def twc_out_2_mcc_action(y: torch.Tensor, fwd_idx: int = 1, rev_idx: int = 0, gain: float = 1.0):
    """
    y: (B, 2) out-layer activations
    Returns: (B, 1) torque in [-1, 1]
    """
    # Signed difference + squashing
    torque = torch.tanh(gain * (y[:, fwd_idx] - y[:, rev_idx])).unsqueeze(1)
    return torque


def twc_out_2_mcc_drive(y: torch.Tensor, fwd_idx: int = 1, rev_idx: int = 0, gain: float = 1.0) -> torch.Tensor:
    """
    y: (B, 2) out-layer activations.
    Returns: (B, 1) unsquashed torque drive (difference between FWD and REV neurons).

    This decoder is intended for policy-gradient methods that model an explicit
    action distribution (e.g. PPO). The returned drive can be interpreted as
    the mean of an unconstrained Gaussian policy that will later be squashed
    with tanh when sampling actions.
    """
    drive = gain * (y[:, fwd_idx] - y[:, rev_idx])
    return drive.unsqueeze(1)
