import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List
from typing import NamedTuple
import torch
from torch.nn import Module


@dataclass
class ObsScale:
    pos_min: float = -1.2
    pos_max: float = 0.6
    vel_min: float = -0.07
    vel_max: float = 0.07


def default_obs_encoder(
    obs: torch.Tensor,
    *,
    n_inputs: int,
    device: torch.device,
    scales: ObsScale = ObsScale(),
    gain: float = 1.0,
) -> torch.Tensor:
    """
    Encode MountainCarContinuous observation [position, velocity]
    into the internal and output states of the sensory neurons of TWC

    Strategy: compute a scalar s_j per input neuron from a linear mix of
    normalized (pos, vel); route positive part to EX, negative part to IN;
    keep GJ at 0 for external drive.

    obs: (batch, 2)
    returns: (batch, 2, n_inputs)
    """
    assert obs.ndim == 2 and obs.shape[1] == 2, "obs must be (batch, 2)"

    # Normalize to approx [-1, 1]
    pos = obs[:, 0]
    vel = obs[:, 1]

    pos_n = 2.0 * (pos - scales.pos_min) / (scales.pos_max - scales.pos_min) - 1.0
    vel_n = 2.0 * (vel - scales.vel_min) / (scales.vel_max - scales.vel_min) - 1.0

    # Simple diverse projections across the 4 sensory neurons
    # If n_inputs != 4, this still works by tiling/truncating weights
    base_W = torch.tensor([
        [1.0, 0.0],   # neuron 0: pos
        [0.0, 1.0],   # neuron 1: vel
        [0.7, 0.7],   # neuron 2: pos+vel
        [1.0, -1.0],  # neuron 3: pos-vel
    ], dtype=obs.dtype, device=obs.device)

    if n_inputs <= base_W.shape[0]:
        W = base_W[:n_inputs]
    else:
        reps = math.ceil(n_inputs / base_W.shape[0])
        W = base_W.repeat(reps, 1)[:n_inputs]

    feats = torch.stack([pos_n, vel_n], dim=1)  # (B, 2)
    s = torch.matmul(feats, W.t()) * gain       # (B, n_inputs)

    ex = torch.clamp(s, min=0.0)
    inh = torch.clamp(-s, min=0.0)
    gj = torch.zeros_like(ex)

    return torch.stack([ex, inh, gj], dim=-1)  # (B, n_inputs, 3)


def default_action_decoder(
    out_state: torch.Tensor,
    *,
    gain: float = 1.0,
) -> torch.Tensor:
    """
    Map output layer out_state (batch, 2) => continuous action in [-1, 1].
    Assumes two output neurons ordered [REV, FWD].
    action = tanh(gain * (FWD - REV)).
    """
    assert out_state.ndim == 2 and out_state.shape[1] == 2, "expected (batch, 2)"
    rev = out_state[:, 0]
    fwd = out_state[:, 1]
    a = torch.tanh(gain * (fwd - rev))
    return a.unsqueeze(-1)  # (batch, 1)

"""
This module now provides only observation encoders and action decoders.
The TWC network built in `utils.twc_builder.build_twc` accepts raw observations
directly and performs encoding internally via the provided decoder.
"""


# -------- Pairwise IN encoder (positive/negative neuron per variable) --------

@dataclass
class InterfacePair:
    obs_index: int
    valleyVal: float
    minVal: float
    maxVal: float
    positive_index: int
    negative_index: int
    minState: float = -1
    maxState: float = 1.0


def make_pairwise_in_encoder(
    pairs: List[InterfacePair],
    *,
    set_inactive_to_min_state: bool = True,
    clamp_to_state_range: bool = True
) -> Callable[[torch.Tensor, int, torch.device], torch.Tensor]:
    """
    Build an observation encoder that mimics the pairwise mapping:
    - For each input variable, choose one of two neurons depending on valleyVal.
    - Drive the selected neuron with a linear mapping from [minVal,maxVal] to [minState,maxState].
    - Optionally drive the inactive neuron to minState (False by default, i.e., no current).

    Returns a callable(obs, n_inputs, device) -> (batch, n_inputs, 3).
    """
    def _standardize_obs(obs, device) -> torch.Tensor:
        """
        Standardize a Gym observation to a batched tensor of shape (B, 2).
        Accepts list/tuple/np.ndarray/torch.Tensor and squeezes singleton dims
        like (2,), (2,1), (1,2,1). Ensures final trailing dim is 2.
        """
        if isinstance(obs, torch.Tensor):
            o = obs
        else:
            o = torch.as_tensor(obs)

        o = o.to(device)
        o = o.squeeze()

        if o.ndim == 1:
            if o.numel() != 2:
                raise ValueError(f"Expected 2 elements in 1D obs, got {o.numel()}")
            o = o.unsqueeze(0)
        elif o.ndim == 2:
            if o.shape[1] != 2 and o.shape[0] == 2:
                o = o.t()
            if o.shape[1] != 2:
                raise ValueError(f"Expected obs shape (B,2), got {tuple(o.shape)}")
        else:
            raise ValueError(f"Unsupported obs ndim {o.ndim}; expected 1 or 2")

        return o
    
    def encoder(obs: torch.Tensor, n_inputs: int, device: torch.device) -> torch.Tensor:
        obs = _standardize_obs(obs, device)
        assert obs.ndim == 2, "obs must be (batch, D)"
        B = obs.shape[0]
        x = torch.zeros(B, 2, n_inputs, dtype=obs.dtype, device=obs.device)

        for p in pairs:
            v = obs[:, p.obs_index]

            mask_pos = v >= p.valleyVal
            mask_neg = ~mask_pos
            rng = (p.maxState - p.minState)


            # Active side uses EX channel; inactive gets zero (or minState if enabled)
            if mask_pos.any():
                cor = v[mask_pos] / p.maxVal
                pot = rng * cor + p.minState
                if clamp_to_state_range:
                    pot = pot.clamp(min=p.minState, max=p.maxState)

                x[mask_pos, 0,p.positive_index] = pot
                x[mask_pos, 1,p.positive_index] = pot

                if set_inactive_to_min_state:
                    x[mask_pos,0, p.negative_index] = p.minState
                    x[mask_pos,1, p.negative_index] = p.minStateinput

            if mask_neg.any():
                cor = v[mask_neg] / (-p.minVal)
                pot = rng * (-cor) + p.minState
                if clamp_to_state_range:
                    pot = pot.clamp(min=p.minState, max=p.maxState)

                x[mask_neg, 0, p.negative_index] = pot
                x[mask_neg, 1, p.negative_index] = pot

                if set_inactive_to_min_state:
                    x[mask_neg, 0, p.positive_index] = p.minState
                    x[mask_neg, 1, p.positive_index] = p.minState

        return x

    return encoder


def mountaincar_pair_encoder(
    *,
    set_inactive_to_min_state: bool = True,
    clamp_to_state_range: bool = True
) -> Callable[[torch.Tensor, int, torch.device], torch.Tensor]:
    """
    Convenience factory using default input order from utils/TWC_fiu.json:
      input = ["PVD"(0), "PLM"(1), "AVM"(2), "ALM"(3)]
    - IN1 (position): positive PLM(1), negative AVM(2), valley -0.3, range [-1.2, 0.6]
    - IN2 (velocity): positive ALM(3), negative PVD(0), valley 0.0,   range [-0.1, 0.1]
    """
    pairs = [
        InterfacePair(obs_index=0, valleyVal=-0.3, minVal=-1.2, maxVal=0.6, positive_index=1, negative_index=2),
        InterfacePair(obs_index=1, valleyVal=0.0,  minVal=-0.07, maxVal=0.07, positive_index=3, negative_index=0),
    ]
    return make_pairwise_in_encoder(pairs, 
                                    set_inactive_to_min_state=False,
                                    clamp_to_state_range=clamp_to_state_range)


# Stochastic decoder intended to be used for actor policy 

class PolicyOut(NamedTuple):
    action: torch.Tensor    # (B,1) sampled action in [-1,1]
    log_prob: torch.Tensor  # (B,1) log π(a|s) with tanh correction
    mean: torch.Tensor      # (B,1) deterministic action = tanh(base_mean)
    std: torch.Tensor       # (B,1) base Normal std (pre-tanh)
    entropy: torch.Tensor   # (B,1) base Normal entropy (proxy)

def stochastic_action_decoder(
    out_state: torch.Tensor,
    *,
    gain: float = 1.0,
    min_log_std: float = -5.0,
    max_log_std: float = 2.0,
    deterministic: bool = False,
    eps: float = 1e-6,
) -> PolicyOut:
    """
    Decode TWC output (B,2) ordered [REV, FWD] into a tanh-squashed Gaussian.

    Base Normal: u ~ N(mu, std), action a = tanh(u).
    log_prob includes change-of-variables: log π(a) = log N(u|mu,std) - log(1 - tanh(u)^2).

    Returns (action, log_prob, mean, std, entropy) with shape (B,1) each.
    """
    assert out_state.ndim == 2 and out_state.shape[1] == 2, "expected (B,2)"
    rev = out_state[:, 0]
    fwd = out_state[:, 1]

    # Unsquashed mean and bounded log-std
    base_mean = gain * (fwd - rev)  # (B,)
    raw = fwd + rev                 # (B,)
    log_std = min_log_std + 0.5 * (torch.tanh(raw) + 1.0) * (max_log_std - min_log_std)
    log_std = torch.clamp(log_std, min_log_std, max_log_std)
    std = torch.exp(log_std)
    var = std * std

    # Deterministic path (evaluation-only)
    mean = torch.tanh(base_mean)
    if deterministic:
        zeros = torch.zeros_like(mean)
        entropy = 0.5 * (1.0 + torch.log(2 * torch.pi * var + eps))
        return PolicyOut(mean.unsqueeze(-1), zeros.unsqueeze(-1), mean.unsqueeze(-1), std.unsqueeze(-1), entropy.unsqueeze(-1))

    # Reparameterized sampling in latent space
    noise = torch.randn_like(base_mean)
    u = base_mean + std * noise          # (B,)
    a = torch.tanh(u)                    # (B,)

    # Base Normal log-prob
    LOG_2PI = 1.8378770664093453  # ln(2π)
    logp_base = -0.5 * (((u - base_mean) ** 2) / (var + eps) + 2.0 * log_std + LOG_2PI)  # (B,)

    # Squash correction: sum over dims; here 1-D so just one term
    squash_correction = torch.log(1.0 - a * a + eps)  # (B,)
    log_prob = (logp_base - squash_correction)        # (B,)

    # Base Normal entropy (proxy)
    entropy = 0.5 * (1.0 + torch.log(2 * torch.pi * var + eps))  # (B,)

    return PolicyOut(a.unsqueeze(-1), 
                     log_prob.unsqueeze(-1), 
                     mean.unsqueeze(-1), 
                     std.unsqueeze(-1), 
                     entropy.unsqueeze(-1))
