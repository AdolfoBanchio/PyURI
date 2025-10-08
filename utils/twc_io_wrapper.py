import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List
from typing import NamedTuple
import torch
import numpy as np
from bindsnet.network import Network
from bindsnet.network.nodes import Input
from bindsnet.network.topology import Connection


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
    into currents for the TWC input layer with 3 channels per neuron (EX, IN, GJ).

    Strategy: compute a scalar s_j per input neuron from a linear mix of
    normalized (pos, vel); route positive part to EX, negative part to IN;
    keep GJ at 0 for external drive.

    obs: (batch, 2)
    returns: (batch, n_inputs, 3)
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

class TwcIOWrapper:
    """
    Wrap a TWC Network (utils.twc_builder.build_TWC()) with sensory Input wiring
    and provide modular obs encoding and action decoding suitable for
    MountainCarContinuous (1-D action in [-1, 1]).

    - Exposes step(obs) -> action and access to last out_state for policy/value heads.

    TODO: Add arguments description.
    """

    def __init__(
        self,
        net: Network,
        *,
        sensory_layer_name: str = "input",
        output_layer_name: str = "output",
        obs_encoder: Optional[Callable[[torch.Tensor, int, torch.device], torch.Tensor]] = None,
        action_decoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.net = net
        self.sensory_layer_name = sensory_layer_name
        self.output_layer_name = output_layer_name
        self.input_name = sensory_layer_name

        self.obs_encoder = obs_encoder
        self.action_decoder = action_decoder


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        # TWC input/output layers
        assert self.sensory_layer_name in net.layers, f"Missing layer '{self.sensory_layer_name}' in TWC net"
        assert self.output_layer_name in net.layers, f"Missing layer '{self.output_layer_name}' in TWC net"

        self.twc_in = net.layers[self.sensory_layer_name]
        self.twc_out = net.layers[self.output_layer_name]

        # Build sensory input layer and identity connection (channel-wise)
        n_inputs = self.twc_in.shape[0]
        assert self.twc_in.shape[1] == 3, "TWC input layer must have 3 channels"
        
        self.net.to(self.device)
        # Debug: print device of relevant torch tensors in the network
        try:
            # Print devices per submodule (layers, connections) considering only local params/buffers
            for mod_name, mod in self.net.named_modules():
                local_devs = []
                for _, p in mod.named_parameters(recurse=False):
                    d = str(p.device)
                    if d not in local_devs:
                        local_devs.append(d)
                for _, b in mod.named_buffers(recurse=False):
                    d = str(b.device)
                    if d not in local_devs:
                        local_devs.append(d)
                if local_devs:
                    label = mod_name if mod_name else 'Network'
                    print(f"[Device] {label}: {', '.join(local_devs)}")
            # Fallback if nothing was found; still show chosen device
            has_any = any(True for _ in self.net.parameters()) or any(True for _ in self.net.buffers())
            if not has_any:
                print(f"[Device] Network default: {self.device}")
        except Exception:
            # Non-fatal: ensure init proceeds even if inspection fails
            print(f"[Device] Network (fallback): {self.device}")
        
        print('--- children ---')
        for n, m in net.named_children():
            print('child:', n, type(m))
        print('--- params ---')
        for n, p in net.named_parameters():
            print('param:', n, p.device)
        # Cache sizes
        self.n_inputs = n_inputs

    def _standardize_obs(self, obs) -> torch.Tensor:
        """
        Standardize a Gym observation to a batched tensor of shape (B, 2).
        Accepts list/tuple/np.ndarray/torch.Tensor and squeezes singleton dims
        like (2,), (2,1), (1,2,1). Ensures final trailing dim is 2.
        """
        if isinstance(obs, torch.Tensor):
            o = obs
        else:
            o = torch.as_tensor(obs)

        o = o.to(self.device, dtype=self.dtype)
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

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_encoder is None:
            return default_obs_encoder(obs.to(self.device, dtype=self.dtype), n_inputs=self.n_inputs, device=self.device)
        # Backward-compat with simple callables expecting (obs, n_inputs, device)
        return self.obs_encoder(obs.to(self.device, dtype=self.dtype), self.n_inputs, self.device)

    def decode_action(self, out_state: torch.Tensor) -> torch.Tensor:
        if self.action_decoder is None:
            return default_action_decoder(out_state)
        return self.action_decoder(out_state)

    def step(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One environment step’s worth of network processing.

        obs: single observation (2,), or batched (B,2), or with extra
             singleton dims like (2,1) / (1,2,1). All are standardized to (B,2).
        returns: out_state: (batch, n_out))

        Step action must be calculated using the self.decode_action
        """
        obs_std = self._standardize_obs(obs)
        x = self.encode_obs(obs_std)  # (B, n_inputs, 3)
        B = x.shape[0]

        # Prepare time-major input for BindsNET: (T=1, B, ...)
        X = x.unsqueeze(0)

        # Make sure batch sizes are correct
        #self.input.set_batch_size(B)
        self.twc_in.set_batch_size(B)
        self.twc_out.set_batch_size(B)

        self.net.run(inputs={self.input_name: X}, time=1, one_step=True)

        # Read out_state from output layer (batch, n_out)
        out_state = self.twc_out.out_state
        return out_state

    def reset(self) -> None:
        """Reset internal states across TWC layers (call at episode boundaries)."""
        self.net.reset_state_variables()


# -------- Pairwise IN encoder (positive/negative neuron per variable) --------

@dataclass
class InterfacePair:
    obs_index: int
    valleyVal: float
    minVal: float
    maxVal: float
    positive_index: int
    negative_index: int
    minState: float = -10.0
    maxState: float = 10.0


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

    def encoder(obs: torch.Tensor, n_inputs: int, device: torch.device) -> torch.Tensor:
        assert obs.ndim == 2, "obs must be (batch, D)"
        B = obs.shape[0]
        x = torch.zeros(B, n_inputs, 3, dtype=obs.dtype, device=obs.device)

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
                
                x[mask_pos, p.positive_index, 0] = pot

                if set_inactive_to_min_state:
                    x[mask_pos, p.negative_index, 0] = p.minState
            
            if mask_neg.any():
                cor = v[mask_neg] / (-p.minVal)
                pot = rng * (-cor) + p.minState
                if clamp_to_state_range:
                    pot = pot.clamp(min=p.minState, max=p.maxState)
                
                x[mask_neg, p.negative_index, 0] = pot

                if set_inactive_to_min_state:
                    x[mask_neg, p.positive_index, 0] = p.minState
            
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
        InterfacePair(obs_index=1, valleyVal=0.0,  minVal=-0.1, maxVal=0.1, positive_index=3, negative_index=0),
    ]
    return make_pairwise_in_encoder(pairs, 
                                    set_inactive_to_min_state=set_inactive_to_min_state,
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
