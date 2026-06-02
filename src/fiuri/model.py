import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import twc.twc_io as io

TWC_JSON = {
  "neurons": {
    "PVD": 0, "PLM": 1, "AVM": 2, "ALM": 3,
    "DVA": 4, "AVD": 5, "PVC": 6, "AVA": 7, "AVB": 8,
    "REV": 9, "FWD": 10
  },

  "groups": {
    "input":  ["PVD", "PLM", "AVM", "ALM"],
    "hidden": ["DVA", "AVD", "PVC", "AVA", "AVB"],
    "output": ["REV", "FWD"]
  },
  "edges": [
    { "src": "PVD", "dst": "DVA", "type": "IN"},
    { "src": "PVD", "dst": "PVC", "type": "IN"},
    { "src": "PVD", "dst": "AVA", "type": "IN"},
    { "src": "PLM", "dst": "DVA", "type": "IN"},
    { "src": "PLM", "dst": "AVD", "type": "IN"},
    { "src": "PLM", "dst": "AVA", "type": "IN"},
    { "src": "PLM", "dst": "PVC", "type": "GJ"},
    { "src": "AVM", "dst": "PVC", "type": "IN"},
    { "src": "AVM", "dst": "AVD", "type": "GJ"},
    { "src": "ALM", "dst": "PVC", "type": "IN"},
    { "src": "ALM", "dst": "AVD", "type": "IN"},
    { "src": "DVA", "dst": "PVC", "type": "IN"},
    { "src": "AVD", "dst": "AVA", "type": "EX"},
    { "src": "AVD", "dst": "AVB", "type": "EX"},
    { "src": "AVD", "dst": "PVC", "type": "EX"},
    { "src": "PVC", "dst": "AVB", "type": "EX"},
    { "src": "PVC", "dst": "AVD", "type": "EX"},
    { "src": "PVC", "dst": "DVA", "type": "EX"},
    { "src": "PVC", "dst": "AVA", "type": "EX"},
    { "src": "AVA", "dst": "AVB", "type": "IN"},
    { "src": "AVA", "dst": "PVC", "type": "IN"},
    { "src": "AVA", "dst": "REV", "type": "EX"},
    { "src": "AVA", "dst": "AVD", "type": "IN"},
    { "src": "AVB", "dst": "FWD", "type": "EX"},
    { "src": "AVB", "dst": "AVA", "type": "IN"},
    { "src": "AVB", "dst": "AVD", "type": "IN"}
  ]
}
# FIU+TWC Class

class PyUriTwc(nn.Module):
    def __init__(self, 
                 config_json: dict, 
                 obs_encoder: callable, 
                 action_decoder: callable,
                 internal_steps: int = 1,
                 device=None):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.action_decoder = action_decoder
        self.internal_steps = internal_steps

        self.neuron_names = config_json['neurons'] 
        self.num_neurons = len(self.neuron_names)
        
        # --- Topology ---
        # Note: Transposed logic handled in MatMul (Batch, Src) @ (Src, Dst) -> (Batch, Dst) requires (Dst, Src) weights
        self.register_buffer('mask_ex', torch.zeros(self.num_neurons, self.num_neurons))
        self.register_buffer('mask_in', torch.zeros(self.num_neurons, self.num_neurons))
        self.register_buffer('mask_gj', torch.zeros(self.num_neurons, self.num_neurons))
        
        for edge in config_json['edges']:
            src = self.neuron_names[edge['src']]
            dst = self.neuron_names[edge['dst']]
            
            if edge['type'] == 'EX': 
                self.mask_ex[dst, src] = 1.0 
            elif edge['type'] == 'IN': 
                self.mask_in[dst, src] = 1.0
            elif edge['type'] == 'GJ': 
                self.mask_gj[dst, src] = 1.0

        self.input_indices = [self.neuron_names[n] for n in config_json['groups']['input']]
        self.output_indices = [self.neuron_names[n] for n in config_json['groups']['output']]

        mask = torch.zeros(self.num_neurons, device=device, dtype=torch.bool)
        mask[self.input_indices] = True
        self.register_buffer('input_bool_mask', mask)
        self.register_buffer('input_idx_base', torch.tensor(self.input_indices))
        
        # --- Parameters ---
        self.weights = nn.Parameter(torch.empty(self.num_neurons, self.num_neurons)) 
        self.thresholds = nn.Parameter(torch.empty(self.num_neurons))                
        self.decay = nn.Parameter(torch.empty(self.num_neurons))                     

        self.sparse_kaiming_uniform()
        nn.init.uniform_(self.thresholds, 0.0, 1.0)
        nn.init.uniform_(self.decay, 0.0, 0.5)

        # State Storage
        self.stored_E = None
        self.stored_O = None

        # TODO: improve the forward_step to be compatible with torch.compile to improve speed.
        self.fast_forward_step = torch.compile(self.forward_step, mode="reduce-overhead")

        #self.fast_forward_step = self.forward_step
        if device:
            self.to(device)


    def sparse_kaiming_uniform(self):
        """
        Initializes weights per-neuron based on the actual number of 
        incoming connections (Effective Fan-In).
        """
        # effective Fan-In for each neuron
        total_mask = self.mask_ex + self.mask_in + self.mask_gj
        fan_in_counts = total_mask.sum(dim=1)

        fan_in_counts[fan_in_counts == 0] = 1.0

        # Calculate Kaiming Bounds per neuron
        gain = math.sqrt(5) # Matches PyTorch default for LeakyReLU/Linear
        bounds = gain * torch.sqrt(3.0 / fan_in_counts)

        with torch.no_grad():
            # U(-1, 1) generator
            raw_noise = (torch.rand_like(self.weights) * 2) - 1 
            
            # Scale weights
            self.weights.data = raw_noise * bounds.unsqueeze(1)

    def set_params_of_name(self, neu_name:str, th, df):
        idx = self.neuron_names[neu_name]
        self.thresholds[idx] = th
        self.decay[idx] = df
        
    def set_params(self, thresholds: dict, decays: dict, weights: dict):
        """
        Set model parameters from dictionaries mapping names to values.
        This is useful for synchronizing with a legacy model.

        Args:
            thresholds (dict): {neuron_name: threshold_val}
            decays (dict): {neuron_name: decay_val}
            weights (dict): {(src_name, dst_name): weight_val}
        """
        with torch.no_grad():
            for name, th in thresholds.items():
                if name in self.neuron_names:
                    idx = self.neuron_names[name]
                    self.thresholds[idx] = float(th)
            
            for name, d in decays.items():
                if name in self.neuron_names:
                    idx = self.neuron_names[name]
                    self.decay[idx] = float(d)

            self.weights.fill_(-100.0)  # Represents ~0 after softplus
            for (src, dst), w in weights.items():
                if src in self.neuron_names and dst in self.neuron_names:
                    src_idx, dst_idx = self.neuron_names[src], self.neuron_names[dst]
                    if w > 1e-6:
                        inv_w = torch.log(torch.expm1(torch.tensor(w, dtype=self.weights.dtype, device=self.weights.device)))
                        self.weights[dst_idx, src_idx] = inv_w

        
    def get_initial_state(self, batch_size, device=None):
        # Initial state is 0.0
        if device is None:
            device = self.weights.device
        return (torch.zeros(batch_size, self.num_neurons, device=device), 
                torch.zeros(batch_size, self.num_neurons, device=device))

    def reset(self, batch_size=1):
        self.stored_E, self.stored_O = self.get_initial_state(batch_size=batch_size)

    def reset_internal_only(self, batch_size=1):
        self.stored_E, _ = self.get_initial_state(batch_size=batch_size)

    def _physics_step(self, state_E, state_O_hybrid):
        """
        Replicates Neuron.py: computeVnext exactly.
        """
        # Chemical Synapses
        W_pos = F.softplus(self.weights)
        W_chem = W_pos * (self.mask_ex - self.mask_in)
        # (Batch, Src) @ (Dst, Src)^T -> (Batch, Dst)
        I_chem = torch.matmul(state_O_hybrid, W_chem.t()) 

        # 2. Gap Junctions
        #   if sourceOut < internalState: currInfluence -= W * sourceOut
        #   elif sourceOut > internalState: currInfluence += W * sourceOut
        #   (else 0)
        E_expanded = state_E.unsqueeze(2)           # (B, Dst, 1)
        O_expanded = state_O_hybrid.unsqueeze(1)    # (B, 1, Src)
        diff = O_expanded - E_expanded
        direction = torch.sign(diff) # returns -1, 0, or 1
        
        # GJ Influence = W * O_src * direction
        W_gj = W_pos * self.mask_gj # get GJ weight connections
        I_gj = (W_gj * O_expanded * direction).sum(dim=2) # Sum over Src

        # Stimulus Calculation
        curr_state = state_E + I_chem + I_gj 

        # Clamp internal states
        curr_state = torch.clamp(curr_state, -10, 10)

        # Nueron Update Rules
        # Condition A: Firing
        # "if currState > self.testThreshold:"
        O_potential = curr_state - self.thresholds
        firing_mask = O_potential > 0.0
        O_new = F.relu(O_potential)
        val_firing = O_new
        
        # Condition B: Decay
        # "elif currState==self.internalstate:" (Implies Influence was 0)
        decay_mask = (~firing_mask) & (torch.abs(state_E - curr_state) < 1e-5)
        val_E_decay = state_E - self.decay
        
        # Condition C: Accumulation (Sub-threshold active)
        val_E_accum = curr_state

        # Combine
        E_new = torch.where(firing_mask, val_firing, torch.where(decay_mask, val_E_decay, val_E_accum))

        return O_new, E_new

    def forward_step(self, obs, state_E, state_O):
        batch_size = obs.shape[0]

        # Encode obs
        input_vals = self.obs_encoder(obs, device=self.weights.device)
             
        idx_tensor = self.input_idx_base.unsqueeze(0).expand(batch_size, -1)        
        zeros = torch.zeros_like(state_E)
        # Create a "Input Only" tensor (zeros elsewhere)
        input_layer_state = zeros.scatter(1, idx_tensor, input_vals)
        
        # Create a "Non-Input Only" tensor
        non_input_mask = ~self.input_bool_mask
        
        state_O_hybrid = (state_O * non_input_mask) + input_layer_state
        state_E_hybrid = (state_E * non_input_mask) + input_layer_state

        E_iter, O_iter = state_E_hybrid, state_O_hybrid
        # Neurons update step (iterated so signal can propagate through the circuit)
        for _ in range(self.internal_steps):
            # re-clamp sensory neurons to input values at each internal step
            E_iter = (E_iter * non_input_mask) + input_layer_state
            O_iter = (O_iter * non_input_mask) + input_layer_state
            O_iter, E_iter = self._physics_step(E_iter, O_iter)
        O_calc, E_calc = O_iter, E_iter
        
        O_new = (O_calc * non_input_mask) + input_layer_state
        E_new = (E_calc * non_input_mask) + input_layer_state

        # Decode motor neurons to action.
        output_neuron_states = E_new[:, self.output_indices]
        action = self.action_decoder(output_neuron_states)

        return action, (E_new, O_new)


    def forward(self, obs):
        """
        Stateful Forward
        """
        obs = obs.to(self.weights.device, non_blocking=True)
        
        if self.stored_E is None or self.stored_E.shape[0] != obs.shape[0]:
            self.stored_E, self.stored_O = self.get_initial_state(obs.shape[0])
            
        action, (new_E, new_O) = self.fast_forward_step(obs, self.stored_E, self.stored_O)
        
        self.stored_E = new_E.detach().clone()
        self.stored_O = new_O.detach().clone()
        return action

    def forward_bptt(self, obs_sequence, initial_state=None):
        device = self.weights.device
        obs_sequence = obs_sequence.to(device)
        B, T, _ = obs_sequence.shape
        
        if initial_state is None:
            E, O = self.get_initial_state(B, device)
        else:
            E, O = initial_state[0].to(device), initial_state[1].to(device)

        actions_list = []
        for t in range(T):
            obs_t = obs_sequence[:, t, :]
            action_t, (new_E, new_O) = self.fast_forward_step(obs_t, E, O)
            E = new_E.clone()
            O = new_O.clone()
            actions_list.append(action_t.clone())

        return torch.stack(actions_list, dim=1), (E, O)


# Surrogate Gradients version
class PyUriTwc_V2(PyUriTwc):
    """
    PyUriTwc with surrogate gradients in the physics step.
    """

    def __init__(self,
                 config_json: dict,
                 obs_encoder: callable,
                 action_decoder: callable,
                 internal_steps: int = 1,
                 steepness_gj: float = 10.0,
                 steepness_fire: float = 10.0,
                 steepness_input: float = 5.0,
                 input_thresh: float = 0.01,
                 device=None):
        # Initialize base architecture/state handling
        super().__init__(config_json=config_json,
                         obs_encoder=obs_encoder,
                         action_decoder=action_decoder,
                         internal_steps=internal_steps,
                         device=device)

        # Surrogate-gradient hyperparameters
        self.steepness_gj = steepness_gj
        self.steepness_fire = steepness_fire
        self.steepness_input = steepness_input
        self.input_thresh = input_thresh

        # TODO: Use a compiled functional step for faster training on GPU 
        self.fast_forward_step = self.forward_step

    def _physics_step(self, state_E, state_O_hybrid):
        """
        Differentiable version of the circuit dynamics using surrogate gates.
        TODO: Improve docstring
        """
        W_pos = F.softplus(self.weights)
        W_chem = W_pos * (self.mask_ex - self.mask_in)
        I_chem = torch.matmul(state_O_hybrid, W_chem.t())

        E_expanded = state_E.unsqueeze(2)
        O_expanded = state_O_hybrid.unsqueeze(1)
        diff = O_expanded - E_expanded
        direction = torch.tanh(self.steepness_gj * diff)
        W_gj = W_pos * self.mask_gj
        I_gj = (W_gj * O_expanded * direction).sum(dim=2)

        curr_state = torch.clamp(state_E + I_chem + I_gj, -10, 10)

        O_potential = curr_state - self.thresholds
        O_new = F.relu(O_potential)
        firing_gate = torch.sigmoid(self.steepness_fire * O_potential)

        diff_influence = torch.abs(state_E - curr_state)
        decay_gate = torch.sigmoid(self.steepness_input * (diff_influence - self.input_thresh))

        val_E_fired = O_potential
        val_E_decay = (state_E - self.decay) + 0.01 * (curr_state - state_E)
        val_E_subthresh = curr_state

        E_nonfired = decay_gate * val_E_subthresh + (1 - decay_gate) * val_E_decay
        E_new = firing_gate * val_E_fired + (1 - firing_gate) * E_nonfired

        return O_new, E_new


class CalibratedActor(nn.Module):
    """
    Wraps a TWC actor with a learnable affine output head.

    ``forward(obs) = base(obs) * action_scale + action_bias`` — the wrapper
    returns the action to execute directly, so engines using it don't need
    a Gaussian/log_std policy layer.

    ``calibrate(env)`` fits ``action_scale`` and ``action_bias`` from a
    random-action rollout so the post-affine policy mean has mean ≈ 0 and
    std ≈ ``target_std`` per dim, with ``action_scale`` capped at
    ``max_scale`` to keep the head from dominating the base actor.
    """

    def __init__(self, base_actor: "PyUriTwc", action_dim: int):
        super().__init__()
        self.base         = base_actor
        self.action_scale = nn.Parameter(torch.ones(action_dim))
        self.action_bias  = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        return self.base(obs) * self.action_scale + self.action_bias

    def forward_bptt(self, obs_sequence, initial_state=None):
        actions, final_state = self.base.forward_bptt(obs_sequence, initial_state)
        return actions * self.action_scale + self.action_bias, final_state

    def reset(self, batch_size: int = 1):
        self.base.reset(batch_size)

    def reset_internal_only(self, batch_size: int = 1):
        self.base.reset_internal_only(batch_size)

    def get_initial_state(self, batch_size: int, device=None):
        return self.base.get_initial_state(batch_size, device)

    @torch.no_grad()
    def calibrate(self, env, n_samples: int = 512,
                  target_std: float = 1.5, max_scale: float = 5.0) -> dict:
        """
        Fit (action_scale, action_bias) on observations from a short
        random-action rollout. Returns a dict of diagnostic scalars.
        """
        import numpy as np  # local import to avoid touching module-level imports
        was_training = self.training
        self.eval()

        obs_buf = []
        obs, _ = env.reset()
        while len(obs_buf) < n_samples:
            obs_buf.append(np.asarray(obs, dtype=np.float32))
            a = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                obs, _ = env.reset()

        device    = self.action_scale.device
        obs_batch = torch.as_tensor(np.stack(obs_buf[:n_samples]),
                                    dtype=torch.float32).to(device)

        self.base.reset(batch_size=obs_batch.shape[0])
        raw = self.base(obs_batch)                # (N, action_dim)
        self.base.reset(batch_size=1)

        raw_mean       = raw.mean(dim=0)
        raw_std        = raw.std(dim=0).clamp_min(1e-3)
        uncapped_scale = (target_std / raw_std)
        new_scale      = uncapped_scale.clamp(max=max_scale)
        new_bias       = -new_scale * raw_mean

        self.action_scale.data.copy_(new_scale.to(self.action_scale.device))
        self.action_bias.data.copy_(new_bias.to(self.action_bias.device))

        if was_training:
            self.train()

        return {
            "calib/raw_mean":       float(raw_mean.abs().mean().item()),
            "calib/raw_std":        float(raw_std.mean().item()),
            "calib/uncapped_scale": float(uncapped_scale.mean().item()),
            "calib/action_scale":   float(new_scale.mean().item()),
            "calib/action_bias":    float(new_bias.mean().item()),
            "calib/scale_capped":   float((uncapped_scale > max_scale).float().mean().item()),
        }


def build_fiuri_twc_mcc():
    return PyUriTwc(config_json=TWC_JSON,
                    obs_encoder=io.mcc_obs_to_potentials,
                    action_decoder=io.twc_out_2_mcc_action)

def build_fiuri_twc_invpen():
    return PyUriTwc(config_json=TWC_JSON,
                    obs_encoder=io.ipen_obs_to_potentials,
                    action_decoder=io.twc_out_2_invpen_action)

