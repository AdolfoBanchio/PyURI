# utils/twc_io_wrapper.py
import torch
import torch.nn.functional as F

# Ranges for MountainCarContinuous
POS_MIN, POS_MAX = -1.2, 0.6
VEL_MAX = 0.07   # symmetric
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

    #PLM_EX_input = torch.where(pos_mask, pos_pot, min_fill)
    #AVM_IN_input = torch.where(pos_mask, min_fill, pos_pot)
    pos_gate = torch.sigmoid((pos - POS_VALLEY_VAL) * SMOOTH_GATE_SHARPNESS)
    PLM_EX_input = pos_gate * pos_pot + (1 - pos_gate) * min_fill
    AVM_IN_input = (1 - pos_gate) * pos_pot + pos_gate * min_fill

    vel_mask = vel >= VEL_VALLEY_VAL
    cor_vel = torch.where(vel_mask, vel / VEL_MAX, vel / -VEL_MAX)
    vel_pot = (MAX_STATE - MIN_STATE) * cor_vel + MIN_STATE
    
    #ALM_EX_input = torch.where(vel_mask, vel_pot, min_fill)
    #PVD_IN_input = torch.where(vel_mask, min_fill, vel_pot)
    vel_gate = torch.sigmoid((vel - VEL_VALLEY_VAL) * SMOOTH_GATE_SHARPNESS)
    ALM_EX_input = vel_gate * vel_pot + (1 - vel_gate) * min_fill
    PVD_IN_input = (1 - vel_gate) * vel_pot + vel_gate * min_fill
    
    zero = torch.zeros_like(pos, device=device)

    ex_in = torch.stack([zero ,PLM_EX_input, zero, ALM_EX_input], dim=1)
    in_in = torch.stack([PVD_IN_input, zero, AVM_IN_input, zero], dim=1)

    return ex_in.to(device), in_in.to(device)

def mcc_obs_encoder_speed_weighted(obs: torch.Tensor, n_inputs=4, device=None, beta: float = 6.0):
    """
    Position/velocity encoder where the relative weight between position and
    velocity drive depends on speed magnitude. When |vel| is small, position
    drive dominates; as |vel| grows, velocity drive dominates.

    obs: (B, 2) -> [position, velocity]
    returns: (ex_in, in_in), each (B, 4)
    """
    if device is None:
        device = obs.device
    pos = obs[:, 0].to(device)
    vel = obs[:, 1].to(device)

    min_fill = torch.full_like(pos, MIN_STATE, device=device)

    # Position mapping with valley correction
    pos_mask = pos >= POS_VALLEY_VAL
    cor_pos = torch.where(pos_mask, pos / POS_MAX, pos / POS_MIN)
    pos_pot = (MAX_STATE - MIN_STATE) * cor_pos + MIN_STATE

    # Velocity mapping with symmetric range
    vel_mask = vel >= VEL_VALLEY_VAL
    cor_vel = torch.where(vel_mask, vel / VEL_MAX, vel / -VEL_MAX)
    vel_pot = (MAX_STATE - MIN_STATE) * cor_vel + MIN_STATE

    # Speed-based weighting in [0,1]
    speed = torch.clamp(torch.abs(vel) / VEL_MAX, 0.0, 1.0)
    w_pos = torch.sigmoid(beta * (1.0 - speed))  # high when slow
    w_vel = 1.0 - w_pos

    # Smooth gates around valley values
    pos_gate = torch.sigmoid((pos - POS_VALLEY_VAL) * SMOOTH_GATE_SHARPNESS)
    vel_gate = torch.sigmoid((vel - VEL_VALLEY_VAL) * SMOOTH_GATE_SHARPNESS)

    # Apply weights to modulate away from MIN_STATE baseline
    pos_mod = MIN_STATE + w_pos * (pos_pot - MIN_STATE)
    vel_mod = MIN_STATE + w_vel * (vel_pot - MIN_STATE)

    PLM_EX_input = pos_gate * pos_mod + (1 - pos_gate) * min_fill
    AVM_IN_input = (1 - pos_gate) * pos_mod + pos_gate * min_fill

    ALM_EX_input = vel_gate * vel_mod + (1 - vel_gate) * min_fill
    PVD_IN_input = (1 - vel_gate) * vel_mod + vel_gate * min_fill

    zero = torch.zeros_like(pos, device=device)
    ex_in = torch.stack([zero, PLM_EX_input, zero, ALM_EX_input], dim=1)
    in_in = torch.stack([PVD_IN_input, zero, AVM_IN_input, zero], dim=1)

    return ex_in.to(device), in_in.to(device)

def mcc_obs_encoder_energy(obs: torch.Tensor, n_inputs=4, device=None, alpha: float = 0.6):
    """
    Energy-based encoder using a blended potential + kinetic heuristic:
      - Potential ~ (sin(3*pos) + 1)/2  in [0,1]
      - Kinetic  ~ (vel/VEL_MAX)^2      in [0,~1]
      - Energy   = alpha*pot + (1-alpha)*kin, then mapped to [MIN_STATE, MAX_STATE]
    Direction is taken from the sign (with smooth gates around valley values).

    obs: (B, 2) -> [position, velocity]
    returns: (ex_in, in_in), each (B, 4)
    """
    if device is None:
        device = obs.device
    pos = obs[:, 0].to(device)
    vel = obs[:, 1].to(device)

    min_fill = torch.full_like(pos, MIN_STATE, device=device)

    # Heuristic energy terms
    height = torch.sin(3.0 * pos)
    pot = (height + 1.0) * 0.5  # [0,1]
    kin = (vel / VEL_MAX) ** 2  # >=0
    energy = torch.clamp(alpha * pot + (1.0 - alpha) * kin, 0.0, 1.0)

    mag_pot = MIN_STATE + (MAX_STATE - MIN_STATE) * energy

    # Smooth gates for direction around valley references
    pos_gate = torch.sigmoid((pos - POS_VALLEY_VAL) * SMOOTH_GATE_SHARPNESS)
    vel_gate = torch.sigmoid((vel - VEL_VALLEY_VAL) * SMOOTH_GATE_SHARPNESS)

    PLM_EX_input = pos_gate * mag_pot + (1 - pos_gate) * min_fill
    AVM_IN_input = (1 - pos_gate) * mag_pot + pos_gate * min_fill

    ALM_EX_input = vel_gate * mag_pot + (1 - vel_gate) * min_fill
    PVD_IN_input = (1 - vel_gate) * mag_pot + vel_gate * min_fill

    zero = torch.zeros_like(pos, device=device)
    ex_in = torch.stack([zero, PLM_EX_input, zero, ALM_EX_input], dim=1)
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


def twc_out_2_mcc_action_tanh(y: torch.Tensor, fwd_idx: int = 1, rev_idx: int = 0, gain: float = 1.0) -> torch.Tensor:
    """
    y: (B, 2) out-layer activations.
    Returns: (B, 1) torque in [-1, 1] with tanh saturation applied to the drive.
    """
    drive = gain * (y[:, fwd_idx] - y[:, rev_idx])
    return torch.tanh(drive).unsqueeze(1)
