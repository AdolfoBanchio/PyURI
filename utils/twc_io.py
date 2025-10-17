# utils/twc_io_wrapper.py
import torch
import torch.nn.functional as F

# Ranges for MountainCarContinuous
POS_MIN, POS_MAX = -1.2, 0.6
VEL_MAX = 0.1   # symmetric
# Min Max states of neurons
MIN_STATE, MAX_STATE = -10,10

POS_VALLEY_VAL = -0.3
VEL_VALLEY_VAL = 0.0
SMOOTH_GATE_SHARPNESS = 8.0

def mcc_obs_encoder(obs: torch.Tensor, n_inputs=4, device=None):
    """
    obs: (B, 2) -> [position, velocity]

    """
    if device is None:
        device = obs.device
    pos = obs[:, 0].to(device)
    vel = obs[:, 1].to(device)

    min_fill = torch.full_like(pos, MIN_STATE, device=device)

    pos_mask = pos >= POS_VALLEY_VAL
    cor_pos = torch.where(pos_mask, pos / POS_MAX, pos / POS_MIN)
    pos_pot = (MAX_STATE - MIN_STATE) * cor_pos + MIN_STATE

    PLM_EX_input = torch.where(pos_mask, pos_pot, min_fill)
    AVM_IN_input = torch.where(pos_mask, min_fill, pos_pot)
    #pos_gate = torch.sigmoid((pos - POS_VALLEY_VAL) * SMOOTH_GATE_SHARPNESS)
    #PLM_EX_input = pos_gate * pos_pot + (1 - pos_gate) * min_fill
    #AVM_IN_input = (1 - pos_gate) * pos_pot + pos_gate * min_fill

    vel_mask = vel >= VEL_VALLEY_VAL
    cor_vel = torch.where(vel_mask, vel / VEL_MAX, vel / -VEL_MAX)
    vel_pot = (MAX_STATE - MIN_STATE) * cor_vel + MIN_STATE
    
    ALM_EX_input = torch.where(vel_mask, vel_pot, min_fill)
    PVD_IN_input = torch.where(vel_mask, min_fill, vel_pot)
    #vel_gate = torch.sigmoid((vel - VEL_VALLEY_VAL) * SMOOTH_GATE_SHARPNESS)
    #ALM_EX_input = vel_gate * vel_pot + (1 - vel_gate) * min_fill
    #PVD_IN_input = (1 - vel_gate) * vel_pot + vel_gate * min_fill
    
    zero = torch.zeros_like(pos, device=device)

    ex_in = torch.stack([zero ,PLM_EX_input, zero, ALM_EX_input], dim=1)
    in_in = torch.stack([PVD_IN_input, zero, AVM_IN_input, zero], dim=1)

    return ex_in.to(device), in_in.to(device)

def twc_out_2_mcc_action(y: torch.Tensor, fwd_idx: int = 1, rev_idx: int = 0, gain: float = 1.0):
    """
    y: (B, 2) out-layer activations
    Returns: (B, 1) torque in [-1, 1]
    """
    neg_St = y[:, rev_idx]
    pos_St = y[:, fwd_idx]
    
    def bounded_affine(xmin, ymin, xmax, ymax, x):
        # affine map [xmin,xmax] -> [ymin,ymax]
        m = (ymax - ymin) / (xmax - xmin)
        b = ymin - m * xmin
        return m * x + b
    
    retval1 = bounded_affine(MIN_STATE, 0, MAX_STATE, 1.0, pos_St)
    retval2 = bounded_affine(MIN_STATE, 0, MAX_STATE, 1.0, neg_St)

    return (retval1 - retval2).unsqueeze(1) * gain


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


def twc_out_2_mcc_action_tanh(y: torch.Tensor, fwd_idx: int = 1, rev_idx: int = 0, gain: float = 1.0) -> torch.Tensor:
    """
    y: (B, 2) out-layer activations.
    Returns: (B, 1) torque in [-1, 1] with tanh saturation applied to the drive.
    """
    drive = gain * (y[:, fwd_idx] - y[:, rev_idx])
    return torch.tanh(drive).unsqueeze(1)
