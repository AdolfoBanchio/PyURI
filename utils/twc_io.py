# utils/twc_io_wrapper.py
import torch
import torch.nn.functional as F

# Ranges for MountainCarContinuous
POS_MIN, POS_MAX = -1.2, 0.6
VEL_MAX = 0.07   # symmetric

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

    # Normalize & soft-limit
    pos_n = torch.clamp(_norm(pos, POS_MIN, POS_MAX), -1.0, 1.0)
    vel_n = torch.clamp(vel / VEL_MAX, -1.0, 1.0)

    # Signed split (positive part to EX, negative magnitude to IN)
    pos_ex = F.relu(pos_n)
    pos_in = F.relu(-pos_n)
    vel_ex = F.relu(vel_n)
    vel_in = F.relu(-vel_n)

    ex_in = torch.stack([pos_ex, pos_in.new_zeros(pos_in.shape), vel_ex, vel_in.new_zeros(vel_in.shape)], dim=1)
    in_in = torch.stack([pos_in, pos_ex.new_zeros(pos_ex.shape), vel_in, vel_ex.new_zeros(vel_ex.shape)], dim=1)

    # If your JSON orders the 4 inputs differently, reorder columns here
    # e.g., ex_in = ex_in[:, [idx0, idx1, idx2, idx3]]
    #       in_in = in_in[:, [idx0, idx1, idx2, idx3]]

    return ex_in.to(device), in_in.to(device)

def twc_out_2_mcc_action(y: torch.Tensor, fwd_idx: int = 0, rev_idx: int = 1, gain: float = 1.0):
    """
    y: (B, 2) out-layer activations
    Returns: (B, 1) torque in [-1, 1]
    """
    # Signed difference + squashing
    torque = torch.tanh(gain * (y[:, fwd_idx] - y[:, rev_idx])).unsqueeze(1)
    return torque

