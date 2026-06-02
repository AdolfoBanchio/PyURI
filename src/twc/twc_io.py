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


# ============================================================
# InvertedPendulum-v5 interface
#   obs = [cart_pos, pole_angle, cart_vel, pole_ang_vel]
#   action in [-3, 3]
# Neuron mapping (per spec):
#   pole_angle >= 0 -> PLM (idx 1) ; pole_angle < 0 -> AVM (idx 2)
#   cart_pos   >= 0 -> ALM (idx 3) ; cart_pos   < 0 -> PVD (idx 0)
#   Action: FWD (idx 1) = push right (+) ; REV (idx 0) = push left (-)
# ============================================================

def ipen_obs_encoder(obs: torch.Tensor, device=None):
    """
    Encode InvertedPendulum-v5 observation to (ex_in, in_in) potentials.

    Supports (B, 4) and (B, T, 4) shapes via `...` indexing.
    Returns two tensors shaped (..., 4) with neuron-index order [PVD, PLM, AVM, ALM].
    Values outside the saturation bounds clamp to ±MAX_STATE.
    """        
    # InvertedPendulum-v5 bounds
    INVPEN_ANGLE_MAX = 0.2       # rad; symmetric (termination at |angle| > 0.2)
    INVPEN_CART_MAX  = 1.0       # m; symmetric saturation for cart position
    if device is None:
        device = obs.device

    cart_pos = obs[..., 0]
    angle    = obs[..., 1]

    min_fill = torch.full_like(cart_pos, MIN_STATE, device=device)
    zero     = torch.zeros_like(cart_pos, device=device)

    # --- Angle encoding (PLM / AVM) ---
    angle_mask = angle >= 0
    # Normalize |angle| into [0, 1] (clamped), then affine map to [MIN_STATE, MAX_STATE]
    cor_angle = torch.clamp(torch.abs(angle) / INVPEN_ANGLE_MAX, 0.0, 1.0)
    angle_pot = (MAX_STATE - MIN_STATE) * cor_angle + MIN_STATE

    PLM_EX_input = torch.where(angle_mask, angle_pot, min_fill)
    AVM_IN_input = torch.where(angle_mask, min_fill, angle_pot)

    # --- Cart position encoding (ALM / PVD) ---
    cart_mask = cart_pos >= 0
    cor_cart = torch.clamp(torch.abs(cart_pos) / INVPEN_CART_MAX, 0.0, 1.0)
    cart_pot = (MAX_STATE - MIN_STATE) * cor_cart + MIN_STATE

    ALM_EX_input = torch.where(cart_mask, cart_pot, min_fill)
    PVD_IN_input = torch.where(cart_mask, min_fill, cart_pot)

    # Stack into (..., 4) with order [PVD, PLM, AVM, ALM]
    ex_in = torch.stack([zero,         PLM_EX_input, zero,         ALM_EX_input], dim=-1)
    in_in = torch.stack([PVD_IN_input, zero,         AVM_IN_input, zero        ], dim=-1)

    return ex_in, in_in


def ipen_obs_to_potentials(obs: torch.Tensor, device=None) -> torch.Tensor:
    """Return potentials [PVD, PLM, AVM, ALM] for InvertedPendulum-v5."""
    ex_in, in_in = ipen_obs_encoder(obs, device=device)
    return ex_in + in_in


def _sign_split_potential(signal: torch.Tensor,
                          saturation: float,
                          device: torch.device):
    """
    Split a signed scalar signal into two positive potentials:
        pos_pot: active (in [MIN_STATE, MAX_STATE]) when signal >= 0, else MIN_STATE
        neg_pot: active (in [MIN_STATE, MAX_STATE]) when signal <  0, else MIN_STATE
 
    The mapping is affine: |signal| / saturation -> [0, 1] -> [MIN_STATE, MAX_STATE].
    Values beyond saturation clamp to MAX_STATE.
 
    Returns (pos_pot, neg_pot) both shaped like `signal`.
    """
    min_fill = torch.full_like(signal, MIN_STATE, device=device)
    pos_mask = signal >= 0.0
 
    magnitude = torch.clamp(torch.abs(signal) / saturation, 0.0, 1.0)
    potential = (MAX_STATE - MIN_STATE) * magnitude + MIN_STATE
 
    pos_pot = torch.where(pos_mask, potential, min_fill)
    neg_pot = torch.where(~pos_mask, potential, min_fill)
 
    return pos_pot, neg_pot

def ipen_obs_to_potentials_v2(obs: torch.Tensor, device=None) -> torch.Tensor:
    """
    Encode InvertedPendulum-v5 obs to a (B, 4) or (B, T, 4) potential tensor.
 
    Neuron order: [PVD=0, PLM=1, AVM=2, ALM=3]
 
    Slot mapping:
        PLM (1): angle >= 0  (excitatory-dominant path to AVD/AVA → FWD)
        AVM (2): angle <  0  (inhibitory path to PVC/AVD → REV bias)
        PVD (0): combined_vel >= 0  (inhibits AVA → reduces REV)
        ALM (3): combined_vel <  0  (excites AVD → increases FWD)
 
    Args:
        obs:    (..., 4) tensor [cart_pos, pole_angle, cart_vel, pole_ang_vel]
        device: target device; defaults to obs.device
 
    Returns:
        (..., 4) potential tensor ready to be injected into PyUriTwc.forward_step
    """
    # InvertedPendulum-v5 saturation bounds
    IPEN_ANGLE_MAX:    float = 0.2    # rad — termination boundary
    IPEN_ANG_VEL_MAX:  float = 5.0   # rad/s — practical saturation
    IPEN_CART_VEL_MAX: float = 5.0   # m/s   — practical saturation
    
    # Blending weights for the velocity composite (must sum to 1.0)
    W_ANG_VEL:  float = 0.7
    W_CART_VEL: float = 0.3

    if device is None:
        device = obs.device
 
    pole_angle   = obs[..., 1]
    cart_vel     = obs[..., 2]
    pole_ang_vel = obs[..., 3]
 
    # ── Angle encoding → PLM / AVM ───────────────────────────────────────────
    PLM_pot, AVM_pot = _sign_split_potential(pole_angle, IPEN_ANGLE_MAX, device)
    # PLM active when angle >= 0  (pole tilts right)
    # AVM active when angle <  0  (pole tilts left)
 
    # ── Velocity composite → PVD / ALM ───────────────────────────────────────
    # Normalise each component to [-1, 1] before blending so their scales
    # are commensurable regardless of the physical units.
    ang_vel_norm  = torch.clamp(pole_ang_vel / IPEN_ANG_VEL_MAX,  -1.0, 1.0)
    cart_vel_norm = torch.clamp(cart_vel     / IPEN_CART_VEL_MAX, -1.0, 1.0)
 
    combined_vel = W_ANG_VEL * ang_vel_norm + W_CART_VEL * cart_vel_norm
    # combined_vel in [-1, 1]; saturation = 1.0 so the affine map is direct
    PVD_pot, ALM_pot = _sign_split_potential(combined_vel, 1.0, device)
    # PVD active when combined_vel >= 0  (pole + cart moving right)
    # ALM active when combined_vel <  0  (pole + cart moving left)
 
    # ── Stack in TWC neuron order [PVD=0, PLM=1, AVM=2, ALM=3] ──────────────
    return torch.stack([PVD_pot, PLM_pot, AVM_pot, ALM_pot], dim=-1)

def ipen_obs_to_potentials_v3(obs: torch.Tensor, device=None) -> torch.Tensor:
    """  
    for each neuron it mimics this encoding:
        corVal = self.value/self.maxValue
        posPot = (self.maxState-self.minState)*corVal+self.minState
        self.sensorialNeuron.setInternalState(posPot)
        self.sensorialNeuron.setOutputState(posPot)
    with:
    - PLM: Pole angle
    - AVM: Pole angular velocity
    - ALM: linear pos
    - PVD: linear velocity
    """
    # Saturation bounds chosen so each obs uses a meaningful slice of the
    # [MIN_STATE, MAX_STATE] potential range during typical play.
    POLE_ANGLE_MAX = 0.2268   # rad — slight margin over the |angle|>0.2 termination
    POLE_VEL_MAX   = 5.0      # rad/s — practical saturation observed during play
    CART_POS_MAX   = 1.0      # m   — matches the |cart_pos|>1 termination boundary
    CART_VEL_MAX   = 5.0      # m/s — practical saturation observed during play

    if device is None:
        device = obs.device

    cart_pos = obs[..., 0]
    pole_angle = obs[..., 1]
    cart_vel = obs[..., 2]
    pole_ang_vel = obs[..., 3]

    def encode_signal(signal: torch.Tensor, saturation: float) -> torch.Tensor:
        cor_val = signal / saturation
        pot = (MAX_STATE - MIN_STATE) * cor_val + MIN_STATE
        return torch.clamp(pot, min=MIN_STATE, max=MAX_STATE)

    PLM_pot = encode_signal(pole_angle, POLE_ANGLE_MAX)
    AVM_pot = encode_signal(pole_ang_vel, POLE_VEL_MAX)
    ALM_pot = encode_signal(cart_pos, CART_POS_MAX)
    PVD_pot = encode_signal(cart_vel, CART_VEL_MAX)

    return torch.stack([PVD_pot, PLM_pot, AVM_pot, ALM_pot], dim=-1)


INVPEN_OUT_MIN   = -3.0      # action saturation (push left)
INVPEN_OUT_MAX   = 3.0       # action saturation (push right)
def twc_out_2_invpen_action(y: torch.Tensor, fwd_idx: int = 1, rev_idx: int = 0, gain: float = 1.0):
    """
    Decode output neuron states to a continuous action in [-3, 3].

    Args:
        y: (B, 2) tensor with output internal states [REV, FWD]
        fwd_idx: FWD index (push right, positive action)
        rev_idx: REV index (push left, negative action)
        gain: optional gain multiplier

    Returns:
        (B, 1) action tensor in [INVPEN_OUT_MIN, INVPEN_OUT_MAX] = [-3, 3]
    """
    neg_St = y[:, rev_idx]
    pos_St = y[:, fwd_idx]

    retval1 = bounded_affine(MIN_STATE, 0.0, MAX_STATE, INVPEN_OUT_MAX, pos_St)
    retval2 = bounded_affine(MIN_STATE, 0.0, MAX_STATE, -INVPEN_OUT_MIN, neg_St)

    return (retval1 - retval2).unsqueeze(1) * gain

INVPEN_PPO_OUT_MIN = -3.0
INVPEN_PPO_OUT_MAX =  3.0
def twc_out_2_invpen_mean(
    y: torch.Tensor,
    fwd_idx: int = 1,
    rev_idx: int = 0,
) -> torch.Tensor:
    """
    Decode TWC output neuron states to a continuous action *mean* in [-3, 3].
 
    Identical arithmetic to ``twc_out_2_invpen_action`` but intended for use
    as the mean of a Gaussian policy inside PPOEngine.  The caller is
    responsible for adding Gaussian noise and computing log-probabilities.
 
    Unlike the TD3 decoder this function does **not** accept a ``gain``
    argument — scaling is handled by the Gaussian std in PPOEngine.
 
    Args:
        y:       (..., 2) tensor with output internal states [..., REV, FWD].
                 Supports any leading batch/time dimensions.
        fwd_idx: Column index of the FWD (push-right) neuron.
        rev_idx: Column index of the REV (push-left) neuron.
 
    Returns:
        (..., 1) action mean tensor in [INVPEN_PPO_OUT_MIN, INVPEN_PPO_OUT_MAX].
    """
    neg_st = y[..., rev_idx]   # REV — push left
    pos_st = y[..., fwd_idx]   # FWD — push right
 
    retval1 = bounded_affine(MIN_STATE, 0.0, MAX_STATE, INVPEN_PPO_OUT_MAX,  pos_st)
    retval2 = bounded_affine(MIN_STATE, 0.0, MAX_STATE, -INVPEN_PPO_OUT_MIN, neg_st)
 
    return (retval1 - retval2).unsqueeze(-1)   # (..., 1)