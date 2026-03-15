# utils/twc_io_wrapper.py
import torch
import torch.nn.functional as F

# Ranges for MountainCarContinuous
POS_MIN, POS_MAX = -1.2, 0.6
VEL_MAX = 0.07   # symmetric
# Min Max states of neurons
MIN_STATE, MAX_STATE = -10, 10

# Interface parameters matching TWFiuriBaseFIU.xml
# IN1: Position interface
POS_VALLEY_VAL = -0.3
POS_MIN_VAL = -1.2
POS_MAX_VAL = 0.6

# IN2: Velocity interface  
VEL_VALLEY_VAL = 0.0
VEL_MIN_VAL = -0.1
VEL_MAX_VAL = 0.1

# OUT1: Action interface
OUT_VALLEY_VAL = 0.0
OUT_MIN_VAL = -1.0
OUT_MAX_VAL = 1.0

# For alternative encoders (smooth gates)
SMOOTH_GATE_SHARPNESS = 8.0

def mcc_obs_to_potentials(obs: torch.Tensor, device=None) -> torch.Tensor:
    """Return potentials [PVD, PLM, AVM, ALM] matching original BinaryInterface.feedNN()."""
    ex_in, in_in = mcc_obs_encoder(obs, device=device)
    return ex_in + in_in

def bounded_affine(xmin: float, ymin: float, xmax: float, ymax: float, x: torch.Tensor) -> torch.Tensor:
    """
    Affine map [xmin, xmax] -> [ymin, ymax] with clamping.
    Matches ariel's BinaryInterface.bounded_affine() exactly.
    """
    a = (ymax - ymin) / (xmax - xmin)
    d = ymin - a * xmin
    y = a * x + d
    y = torch.clamp(y, min=ymin, max=ymax)
    return y

def mcc_obs_encoder(obs: torch.Tensor, device=None):
    """
    Soporta (B, 2) y (B, T, 2) preservando las dimensiones.
    """
    if device is None:
        device = obs.device
    
    # Usamos ellipsis (...) para capturar dimensiones previas (Batch o Batch+Time)
    pos = obs[..., 0]
    vel = obs[..., 1]

    min_fill = torch.full_like(pos, MIN_STATE, device=device)
    zero = torch.zeros_like(pos, device=device)

    # --- Codificacion de Posicion (PLM/AVM) ---
    pos_mask = pos >= POS_VALLEY_VAL
    cor_pos = torch.where(pos_mask, pos / POS_MAX_VAL, pos / (-POS_MIN_VAL))
    pos_pot = (MAX_STATE - MIN_STATE) * cor_pos + MIN_STATE
    neg_pos_pot = (MAX_STATE - MIN_STATE) * (-cor_pos) + MIN_STATE

    PLM_EX_input = torch.where(pos_mask, pos_pot, min_fill)
    AVM_IN_input = torch.where(pos_mask, min_fill, neg_pos_pot)

    # --- Codificacion de Velocidad (ALM/PVD) ---
    vel_mask = vel >= VEL_VALLEY_VAL
    cor_vel = torch.where(vel_mask, vel / VEL_MAX_VAL, vel / (-VEL_MIN_VAL))
    vel_pot = (MAX_STATE - MIN_STATE) * cor_vel + MIN_STATE
    neg_vel_pot = (MAX_STATE - MIN_STATE) * (-cor_vel) + MIN_STATE

    ALM_EX_input = torch.where(vel_mask, vel_pot, min_fill)
    PVD_IN_input = torch.where(vel_mask, min_fill, neg_vel_pot)

    # Stack en la última dimensión para mantener (B, 4) o (B, T, 4)
    ex_in = torch.stack([zero, PLM_EX_input, zero, ALM_EX_input], dim=-1)
    in_in = torch.stack([PVD_IN_input, zero, AVM_IN_input, zero], dim=-1)

    return ex_in, in_in

def twc_out_2_mcc_action(y: torch.Tensor, fwd_idx: int = 1, rev_idx: int = 0, gain: float = 1.0):
    """
    Decodes output neuron states to action, exactly matching ariel's BinaryInterface.getFeedBackNN().
    
    This matches the behavior of OUT1 interface:
    - FWD (positive neuron) and REV (negative neuron) -> scalar action
    
    Args:
        y: (B, 2) tensor with output layer internal states [REV, FWD]
        fwd_idx: index of FWD neuron (default 1)
        rev_idx: index of REV neuron (default 0)
        gain: optional gain multiplier
        
    Returns:
        (B, 1) tensor with action in [-1, 1]
    """
    # Get internal states (matching ariel: uses getInternalState())
    neg_St = y[:, rev_idx]  # REV internal state
    pos_St = y[:, fwd_idx]   # FWD internal state
    
    # Map from [minState, maxState] to [0, maxValue] and [0, -minValue]
    retval1 = bounded_affine(MIN_STATE, 0.0, MAX_STATE, OUT_MAX_VAL, pos_St)
    retval2 = bounded_affine(MIN_STATE, 0.0, MAX_STATE, -OUT_MIN_VAL, neg_St)
    
    # Return difference (matching ariel: retVal1 - retVal2)
    return (retval1 - retval2).unsqueeze(1) * gain
