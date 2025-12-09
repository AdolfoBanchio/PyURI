import torch
import torch.nn as nn
import torch.nn.functional as F
import json

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
 
MIN_STATE_INTERFACE = -10.0
MAX_STATE_INTERFACE = 10.0

# Neuron.py clamping limits
NEURON_MIN_CLIP = -10.0
NEURON_MAX_CLIP = 10.0

# Environment Bounds (MountainCarContinuous)
POS_MIN_VAL, POS_MAX_VAL, POS_VALLEY_VAL = -1.2, 0.6, -0.3   # Note: Fixed POS_MIN_VAL from 1.2 to -1.2
VEL_MIN_VAL, VEL_MAX_VAL, VEL_VALLEY_VAL = -0.1, 0.1, 0.0  # Note: Fixed VEL_MIN_VAL from 0.07 to -0.07

# Output scaling
OUT_MIN_VAL = 1.0
OUT_MAX_VAL = 1.0

# ==========================================
# 2. Exact Replication of Encoders/Decoders
# ==========================================

def bounded_affine(xmin, ymin, xmax, ymax, x):
    """Matches ModelInterfaces.py: bounded_affine"""
    if abs(xmax - xmin) < 1e-6: 
        return torch.clamp(x, min=ymin, max=ymax)
    
    a = (ymax - ymin) / (xmax - xmin)
    d = ymin - a * xmin
    y = a * x + d
    
    return torch.clamp(y, min=ymin, max=ymax)

def mcc_obs_encoder(obs: torch.Tensor, device=None):
    """
    Matches ModelInterfaces.py: BinaryInterface.feedNN()
    Maps Env Obs -> [-20, 20] Potentials
    """
    if device is None: device = obs.device
    if obs.dim() == 1: obs = obs.unsqueeze(0)
    
    pos = obs[:, 0]
    vel = obs[:, 1]
    
    # Fill values
    min_fill = torch.full_like(pos, MIN_STATE_INTERFACE, device=device)
    zero_fill = torch.zeros_like(pos, device=device) # Not strictly used as 'zero', but as placeholders

    # --- Position (IN1) ---
    pos_mask = pos >= POS_VALLEY_VAL
    
    # Logic from feedNN:
    # If >= Valley: corVal = value/max; Pot = (Max-Min)*corVal + Min
    # If < Valley:  corVal = value/(-min); Pot = (Max-Min)*(-corVal) + Min
    cor_pos = torch.where(pos_mask, pos / POS_MAX_VAL, pos / (-POS_MIN_VAL))
    
    # Calculate Potential
    term_range = MAX_STATE_INTERFACE - MIN_STATE_INTERFACE # 40.0
    
    pos_pot = torch.where(
        pos_mask,
        term_range * cor_pos + MIN_STATE_INTERFACE,      # Positive Branch
        term_range * (-cor_pos) + MIN_STATE_INTERFACE    # Negative Branch
    )
    
    # Mapping to Neurons (PLM/AVM)
    # IN1: position -> PLM (positive) / AVM (negative)
    PLM = torch.where(pos_mask, pos_pot, min_fill)
    AVM = torch.where(pos_mask, min_fill, pos_pot)

    # --- Velocity (IN2) ---
    vel_mask = vel >= VEL_VALLEY_VAL
    
    cor_vel = torch.where(vel_mask, vel / VEL_MAX_VAL, vel / (-VEL_MIN_VAL))
    
    vel_pot = torch.where(
        vel_mask,
        term_range * cor_vel + MIN_STATE_INTERFACE,
        term_range * (-cor_vel) + MIN_STATE_INTERFACE
    )
    
    # Mapping to Neurons (ALM/PVD)
    # IN2: velocity -> ALM (positive) / PVD (negative)
    ALM = torch.where(vel_mask, vel_pot, min_fill)
    PVD = torch.where(vel_mask, min_fill, vel_pot)

    # Stack Order: [PVD, PLM, AVM, ALM] matching JSON group "input"
    # Note: min_fill is -20.0. The inputs are FORCED to -20.0 if inactive.
    # We return one tensor for the values to clamp
    input_vals = torch.stack([PVD, PLM, AVM, ALM], dim=1)
    
    return input_vals

def twc_out_2_mcc_action(y: torch.Tensor, fwd_idx=1, rev_idx=0, gain=1.0):
    neg_St = y[:, rev_idx] 
    pos_St = y[:, fwd_idx] 
    
    # FIX: Map both to POSITIVE [0, 1] range
    # In legacy, minVal is -1.0. bounded_affine uses -minVal = 1.0.
    retval1 = bounded_affine(MIN_STATE_INTERFACE, 0.0, MAX_STATE_INTERFACE, OUT_MAX_VAL, pos_St)
    retval2 = bounded_affine(MIN_STATE_INTERFACE, 0.0, MAX_STATE_INTERFACE, OUT_MIN_VAL, neg_St) # Use Positive Target
    
    # Then Subtract: FWD - REV
    return (retval1 - retval2).unsqueeze(1) * gain
# ==========================================
# 3. The GPU-Optimized Circuit Class
# ==========================================

class PyUriTwc(nn.Module):
    def __init__(self, 
                 config_json: dict, 
                 obs_encoder: callable, 
                 action_decoder: callable, 
                 device=('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.device = device
        self.obs_encoder = obs_encoder
        self.action_decoder = action_decoder
        
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
            
            # Legacy Connection.py: ChemEx, ChemIn, AGJ/SGJ
            # Legacy Neuron.py simplifies this to Ex(+), In(-), Else(GJ)
            if edge['type'] == 'EX': 
                self.mask_ex[dst, src] = 1.0 
            elif edge['type'] == 'IN': 
                self.mask_in[dst, src] = 1.0
            elif edge['type'] == 'GJ': 
                self.mask_gj[dst, src] = 1.0

        self.input_indices = [self.neuron_names[n] for n in config_json['groups']['input']]
        self.output_indices = [self.neuron_names[n] for n in config_json['groups']['output']]

        # --- Parameters ---
        # Legacy Model.py: noiseParam limits weights/thresholds/decay to [0.0, 10.0]
        # We initialize them in a biologically plausible range for this model
        self.weights = nn.Parameter(torch.empty(self.num_neurons, self.num_neurons)) # Range 0-5
        self.thresholds = nn.Parameter(torch.empty(self.num_neurons))                # Range 0-5
        self.decay = nn.Parameter(torch.empty(self.num_neurons))                     # Range 0-0.5

        nn.init.kaiming_uniform_(self.weights)
        nn.init.uniform_(self.thresholds, 0.0, 1.0)
        nn.init.uniform_(self.decay, 0.0, 0.5)

        # State Storage
        self.stored_E = None
        self.stored_O = None

    def get_initial_state(self, batch_size, device):
        # Initial state is 0.0 (Quiescent)
        return torch.zeros(batch_size, self.num_neurons, device=device), torch.zeros(batch_size, self.num_neurons, device=device)

    def reset_state(self, batch_size=1):
        self.stored_E, self.stored_O = self.get_initial_state(batch_size, self.device)

    def reset_internal_only(self, batch_size=1):
        self.stored_E, _ = self.get_initial_state(batch_size, self.device)

    def _physics_step(self, state_E, state_O_hybrid):
        """
        Replicates Neuron.py: computeVnext exactly.
        """
        # 1. Chemical Synapses
        # Legacy: currInfluence += W * sourceOut (Ex)
        # Legacy: currInfluence -= W * sourceOut (In)
        W_pos = F.softplus(self.weights)
        W_chem = W_pos * (self.mask_ex - self.mask_in)
        # (Batch, Src) @ (Dst, Src)^T -> (Batch, Dst)
        I_chem = torch.matmul(state_O_hybrid, W_chem.t()) 

        # 2. Gap Junctions (The "Else" block in Neuron.py)
        # Legacy Logic: 
        #   if sourceOut < internalState: currInfluence -= W * sourceOut
        #   elif sourceOut > internalState: currInfluence += W * sourceOut
        #   (else 0)
        
        E_expanded = state_E.unsqueeze(2)           # (B, Dst, 1)
        O_expanded = state_O_hybrid.unsqueeze(1)    # (B, 1, Src)
        
        # Direction Logic:
        # +1 if O_src > E_dst
        # -1 if O_src < E_dst
        # 0 if O_src == E_dst (Handled by sign)
        
        # Note: We use O_expanded - E_expanded to check relation
        diff = O_expanded - E_expanded
        direction = torch.sign(diff) # returns -1, 0, or 1
        
        # Influence = W * O_src * direction
        # Note: Legacy multiplies by O_src, not the difference (O-E). 
        # "currInfluence = currInfluence - dc.getTestWeight() * sourceOutState"
        W_gj = W_pos * self.mask_gj
        I_gj = (W_gj * O_expanded * direction).sum(dim=2) # Sum over Src

        # 3. Stimulus Calculation
        # Legacy: currState = internalstate + currInfluence
        # Note: No external current added here, it was injected via state_O_hybrid clamping
        curr_state = state_E + I_chem + I_gj 

        # 4. Hard Clamping (Neuron.py)
        # "if currState < -10: ... elif currState > 10: ..."
        curr_state = torch.clamp(curr_state, -10, 10)

        # 5. Update Rules (Neuron.py if/elif/else block)
        
        # Condition A: Firing
        # "if currState > self.testThreshold:"
        # Output = currState - threshold
        # Internal = currState - threshold
        firing_mask = curr_state > self.thresholds
        val_firing = curr_state - self.thresholds
        
        # Condition B: Decay
        # "elif currState==self.internalstate:" (Implies Influence was 0)
        # We use a small epsilon for float equality check
        net_influence = I_chem + I_gj
        decay_mask = (~firing_mask) & (torch.abs(state_E - curr_state) < 1e-5)
        
        val_E_decay = state_E - self.decay
        val_O_decay = torch.zeros_like(state_E) # "self.bufferedOutputState=0"
        
        # Condition C: Accumulation (Sub-threshold active)
        # "else:"
        # Internal = currState
        # Output = 0
        val_E_accum = curr_state
        val_O_accum = torch.zeros_like(state_E)

        # Combine
        E_new = torch.where(firing_mask, val_firing, torch.where(decay_mask, val_E_decay, val_E_accum))
        O_new = torch.where(firing_mask, val_firing, torch.where(decay_mask, val_O_decay, val_O_accum))

        return O_new, E_new

    def forward_step(self, obs, state_E, state_O):
        batch_size = obs.shape[0]

        # 1. Encode
        input_vals = self.obs_encoder(obs, device=self.device)
        
        # 2. Hybrid State Construction
        state_O_hybrid = state_O.clone() 
        state_O_hybrid[:, self.input_indices] = input_vals
        
        state_E_hybrid = state_E.clone()
        state_E_hybrid[:, self.input_indices] = input_vals

        # 3. Physics
        O_new, E_new = self._physics_step(state_E_hybrid, state_O_hybrid)

        # 4. Input Persistence (FIX: Uncommented for exact replication)
        # Force the inputs to hold their encoded value for the next step's "old state"
        O_new[:, self.input_indices] = input_vals
        E_new[:, self.input_indices] = input_vals

        # 5. Decode
        output_neuron_states = E_new[:, self.output_indices]
        action = self.action_decoder(output_neuron_states)

        return action, (E_new, O_new)

    def forward(self, obs):
        """Stateful Forward"""
        if self.stored_E is None or self.stored_E.shape[0] != obs.shape[0]:
            self.reset_state(obs.shape[0])
            
        action, (new_E, new_O) = self.forward_step(obs, self.stored_E, self.stored_O)
        
        self.stored_E = new_E.detach()
        self.stored_O = new_O.detach()
        return action

    def forward_bptt(self, obs_sequence, initial_state=None):
        B, T, _ = obs_sequence.shape
        if initial_state is None:
            E, O = self.get_initial_state(B, self.device)
        else:
            E, O = initial_state

        actions_list = []
        for t in range(T):
            obs_t = obs_sequence[:, t, :]
            action_t, (E, O) = self.forward_step(obs_t, E, O)
            actions_list.append(action_t)

        return torch.stack(actions_list, dim=1), (E, O)


# Surrogate Gradients version

class PyUriTwc_V2(nn.Module):
    def __init__(self, 
                 config_json: dict, 
                 obs_encoder: callable, 
                 action_decoder: callable,
                 # surrogate gradients parameters
                 steepness_gj: float = 10.0,
                 steepness_fire: float = 10.0,
                 steepness_input: float = 5.0,
                 input_thresh: float = 0.01,
                 leaky_slope: float = 0.01,
                 device=('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.device = device
        self.obs_encoder = obs_encoder
        self.action_decoder = action_decoder
        
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
            
            # Legacy Connection.py: ChemEx, ChemIn, AGJ/SGJ
            # Legacy Neuron.py simplifies this to Ex(+), In(-), Else(GJ)
            if edge['type'] == 'EX': 
                self.mask_ex[dst, src] = 1.0 
            elif edge['type'] == 'IN': 
                self.mask_in[dst, src] = 1.0
            elif edge['type'] == 'GJ': 
                self.mask_gj[dst, src] = 1.0

        self.input_indices = [self.neuron_names[n] for n in config_json['groups']['input']]
        self.output_indices = [self.neuron_names[n] for n in config_json['groups']['output']]

        # --- Parameters ---
        # Legacy Model.py: noiseParam limits weights/thresholds/decay to [0.0, 10.0]
        # We initialize them in a biologically plausible range for this model
        self.weights = nn.Parameter(torch.rand(self.num_neurons, self.num_neurons) * 5.0) # Range 0-5
        self.thresholds = nn.Parameter(torch.rand(self.num_neurons) * 5.0)                # Range 0-5
        self.decay = nn.Parameter(torch.rand(self.num_neurons) * 0.5)                     # Range 0-0.5

        # surrogate gradients parameters
        self.steepness_gj = steepness_gj
        self.steepness_fire = steepness_fire
        self.steepness_input = steepness_input
        self.input_thresh = input_thresh
        self.leaky_slope = leaky_slope

        # State Storage
        self.stored_E = None
        self.stored_O = None

    def get_initial_state(self, batch_size, device):
        # Initial state is 0.0 (Quiescent)
        return torch.zeros(batch_size, self.num_neurons, device=device), torch.zeros(batch_size, self.num_neurons, device=device)

    def reset_state(self, batch_size=1):
        self.stored_E, self.stored_O = self.get_initial_state(batch_size, self.device)

    def reset_internal_only(self, batch_size=1):
        self.stored_E, _ = self.get_initial_state(batch_size, self.device)

    def _physics_step(self, state_E, state_O_hybrid):
        """
        Replicates Neuron.py: computeVnext exactly.
        """
        # 1. Chemical Synapses
        # Legacy: currInfluence += W * sourceOut (Ex)
        # Legacy: currInfluence -= W * sourceOut (In)
        W_pos = F.softplus(self.weights)
        W_chem = W_pos * (self.mask_ex - self.mask_in)
        # (Batch, Src) @ (Dst, Src)^T -> (Batch, Dst)
        I_chem = torch.matmul(state_O_hybrid, W_chem.t()) 

        # 2. Gap Junctions (The "Else" block in Neuron.py)
        # Legacy Logic: 
        #   if sourceOut < internalState: currInfluence -= W * sourceOut
        #   elif sourceOut > internalState: currInfluence += W * sourceOut
        #   (else 0)
        E_expanded = state_E.unsqueeze(2)           # (B, Dst, 1)
        O_expanded = state_O_hybrid.unsqueeze(1)    # (B, 1, Src)
        # Direction Logic:
        # instead of sign, use tanh function for flowing gradients
        # Note: We use O_expanded - E_expanded to check relation
        diff = O_expanded - E_expanded
        direction = torch.tanh(self.steepness_gj * diff) # returns -1, 0, or 1
        # Influence = W * O_src * direction
        W_gj = W_pos * self.mask_gj
        I_gj = (W_gj * O_expanded * direction).sum(dim=2) # Sum over Src

        # 3. Stimulus Calculation
        # Legacy: currState = internalstate + currInfluence
        curr_state = state_E + I_chem + I_gj 

        # 4. Hard Clamping (Neuron.py)
        curr_state = torch.clamp(curr_state, -10, 10)

        # 5. Update Rules (Neuron.py if/elif/else block)
        # instead of mask cases, will use mask 'gates' to let gradients flow
        # Condition A: Firing
        # "if currState > self.testThreshold:"
        # Output = currState - threshold
        # Internal = currState - threshold
        O_new = F.leaky_relu(curr_state - self.thresholds, negative_slope=self.leaky_slope)
        firing_gate = torch.sigmoid(self.steepness_fire * (curr_state - self.thresholds))
        
        # Condition B: Decay
        # "elif currState==self.internalstate:" (Implies Influence was 0)
        # We use a small epsilon for float equality check
        diff = torch.abs(state_E - curr_state)
        #  ~0 if S == E, y ~1 if S != E.
        # to decide betwen elif/else block for internal state
        decay_gate = torch.sigmoid(self.steepness_input * (diff - self.input_thresh))
        
        val_E_fired = O_new 
        val_E_decay = state_E - self.decay
        val_E_subthresh = curr_state
        
        # Evaluate final E value for nonfiring case using gate
        E_nonfired = decay_gate * val_E_subthresh + (1 - decay_gate) * val_E_decay

        # Combine
        E_new = firing_gate * val_E_fired + (1 - firing_gate) * E_nonfired

        return O_new, E_new

    def forward_step(self, obs, state_E, state_O):
        batch_size = obs.shape[0]

        # 1. Encode
        input_vals = self.obs_encoder(obs, device=self.device)
        
        # 2. Hybrid State Construction
        state_O_hybrid = state_O.clone() 
        state_O_hybrid[:, self.input_indices] = input_vals
        
        state_E_hybrid = state_E.clone()
        state_E_hybrid[:, self.input_indices] = input_vals

        # 3. Physics
        O_new, E_new = self._physics_step(state_E_hybrid, state_O_hybrid)

        # 4. Input Persistence (FIX: Uncommented for exact replication)
        # Force the inputs to hold their encoded value for the next step's "old state"
        O_new[:, self.input_indices] = input_vals
        E_new[:, self.input_indices] = input_vals

        # 5. Decode
        output_neuron_states = E_new[:, self.output_indices]
        action = self.action_decoder(output_neuron_states)

        return action, (E_new, O_new)

    def forward(self, obs):
        """Stateful Forward"""
        if self.stored_E is None or self.stored_E.shape[0] != obs.shape[0]:
            self.reset_state(obs.shape[0])
            
        action, (new_E, new_O) = self.forward_step(obs, self.stored_E, self.stored_O)
        
        self.stored_E = new_E.detach()
        self.stored_O = new_O.detach()
        return action

    def forward_bptt(self, obs_sequence, initial_state=None):
        B, T, _ = obs_sequence.shape
        if initial_state is None:
            E, O = self.get_initial_state(B, self.device)
        else:
            E, O = initial_state

        actions_list = []
        for t in range(T):
            obs_t = obs_sequence[:, t, :]
            action_t, (E, O) = self.forward_step(obs_t, E, O)
            actions_list.append(action_t)

        return torch.stack(actions_list, dim=1), (E, O)
    
def build_fiuri_twc():
    return PyUriTwc(config_json=TWC_JSON,
                    obs_encoder=mcc_obs_encoder,
                    action_decoder=twc_out_2_mcc_action)

def build_fiuri_twc_v2():
    return PyUriTwc_V2(config_json=TWC_JSON,
                       obs_encoder=mcc_obs_encoder,
                       action_decoder=twc_out_2_mcc_action,
                       steepness_gj = 10.0,
                       steepness_fire = 10.0,
                       steepness_input = 5.0,
                       input_thresh = 0.01,
                       leaky_slope = 0.01,)