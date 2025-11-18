from typing import Optional
import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F

# Optional acceleration
try:
    from torch_scatter import scatter_add as _scatter_add_fast
except Exception:
    _scatter_add_fast = None


def _scatter_add_batched(src_be: torch.Tensor, dst_e: torch.Tensor, out_bn: torch.Tensor) -> torch.Tensor:
    """
    Batched scatter_add into postsyn indices.

    src_be: (B, E)  contributions per edge (already multiplied by weights, and sign if any)
    dst_e:  (E,)    integer dst neuron indices
    out_bn: (B, N)  output buffer to accumulate into (will be updated in-place)
    """
    if _scatter_add_fast is not None:
        idx = dst_e.expand(src_be.size(0), -1)    # (B, E)
        _scatter_add_fast(src_be, idx, dim=1, out=out_bn)
    else:
        idx = dst_e.unsqueeze(0).expand(src_be.size(0), -1)  # (B, E)
        out_bn.scatter_add_(1, idx, src_be)
    return out_bn


class FIURIModule(nn.Module):
    """  
    Nerual layer of the FIURI model. (fully pytorch)
    Sn = En + sum(I_jin) j=1..m where m is ammount of connections(9) 
         
    Ij in = 
            ωj * Oj if Oj ≥ En y gap junct. 
            -ωj * Oj if Oj < En y gap junct. 
            ωj * Oj chemical excitatory 
            -ωj * Oj chemical inhibitory
    where Oj is the output state of the presynaptic neuron j and ωj is the weight of the connection from neuron j to neuron n.
    
    On = ( Sn- Tn if Sn > Tn 
         ( 0 other case (10) 
    
        | Sn - Tn if Sn > Tn
    En =  En - dn if Sn ≤ Tn and Sn = En  (11) 
        \ Sn other case  
    
    where:
        - En and On represent the internal state and the output state of neuron n, respectively. (not learnable)
        - Sn represents the stimiulus comming through the connections due to the currents Ij_in.
        - Eqs 10,11 represent the dynamic of the neuron.
        - Tn and dn are the neuronal parameters that have to be learned and represent:
            - Tn the firing threshold
            - dn the decay factor (due to not enough stimulus)
    """
    def __init__(
        self,
        num_cells: Optional[int] = None,
        initial_in_state: Optional[float] = 0.0,   # scalar default
        initial_out_state: Optional[float] = 0.0,  # scalar default
        initial_threshold: Optional[float] = 1.0,
        initial_decay: Optional[float] = 0.1,
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
        rnd_init: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert (num_cells is not None), "Must provide number of neurons per layer"      
        # save num of cells
        self.num_cells = num_cells

        self.threshold = nn.Parameter(torch.empty(num_cells), requires_grad=True) 
        self.decay     = nn.Parameter(torch.empty(num_cells), requires_grad=True)
        
        if rnd_init:
            # TODO: Check and define proper random init ranges
            nn.init.uniform_(self.threshold, -1.0, 2.0)
            nn.init.uniform_(self.decay, 0.05, 1.0)
        else:
            nn.init.constant_(self.threshold, initial_threshold)
            nn.init.constant_(self.decay, initial_decay)
        
        self._init_E = float(initial_in_state)
        self._init_O = float(initial_out_state)

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max


    def _compute_gj_sum(self, gj_bundle, o_pre: torch.Tensor, current_in_state: torch.Tensor, out_buffer: torch.Tensor) -> None:
        """
        Computes GJ sum and scatters it IN-PLACE into out_buffer.
        """
        src, dst, w = gj_bundle
        if src.numel() == 0:
            return  # No-op, out_buffer is unchanged

        B = o_pre.size(0)
        Oj = o_pre[:, src]              # (B, E_gj)
        En = current_in_state[:, dst]   # (B, E_gj)
        
        sgn = torch.sign(Oj - En) # -1, 0, or 1
        #sgn = torch.where(Oj >= En, Oj.new_ones(Oj.shape), -Oj.new_ones(Oj.shape))
        contrib = Oj * w * sgn          # (B, E_gj)

        _scatter_add_batched(contrib, dst, out_buffer)
        # No return value, as out_buffer is modified in-place

    def _calculate_new_state(self, S, E, T, D):
        """
        Calculates new (E, O) state from S, E, T, and D.
        """
        # Match ariel's exact equality check: currState==self.internalstate
        eqE = S == E
        gt = S > T
        mask = (~gt) & eqE

        new_o = F.relu(S - T)
        new_e = torch.where(S > T, new_o, torch.where(mask, E - D, S))
        return new_o, (new_e, new_o)

    def neuron_step(self, state):
        """
        Performs one step of the neuron dynamics.
        If S is None, uses S = in_state (i.e., as if input_current were zero).
        """
        E, O = state if state else (self._init_E, self._init_O)
        S = torch.clamp(E, self.clamp_min, self.clamp_max)

        return self._calculate_new_state(S, E, self.threshold, self.decay)

    def forward(self, chem_influence, state = None, gj_bundle = None, o_pre = None) -> torch.Tensor:
        B = chem_influence.size(0)
        
        # Extract current state for use in gap junction computation
        E, O = state if state else (self._init_E, self._init_O)
        
        if gj_bundle is not None:
            assert o_pre is not None, "o_pre must be provided when gj_bundle is not None"
            current_E_tensor = E if isinstance(E, torch.Tensor) else torch.full((B, self.num_cells), E, dtype=chem_influence.dtype, device=chem_influence.device)
            
            self._compute_gj_sum(
                gj_bundle, 
                o_pre, 
                current_in_state=current_E_tensor, 
                out_buffer=chem_influence # in-place modification
            )
        
        # Note: E is still the *original* state[0]
        S = torch.clamp(E + chem_influence, self.clamp_min, self.clamp_max)

        return self._calculate_new_state(S, E, self.threshold, self.decay)
    

class FiuriDenseConn(nn.Module):
    """  
        This class is intended to be used to connect FIURI layers with EX/IN conns

        Contains a dense graph of the edges between the pre and post synaptic layers
        and also a mask of the edges that must not be taken into account.

        params:
            - n_pre, n_post: cardinality of pre and post synaptic layers 
            - w_mask: tensor of size pre x post setting wich weights wont be updated
            - type: "EX" or "IN", will define the sign of the wheigt on the forward passs.
    """
    def __init__(self, n_pre: int, 
                n_post: int, 
                w_mask: torch.Tensor, # (n_pre, n_poost) shape
                type: str):
        super().__init__()
        if type not in ["EX","IN"]:
            raise ValueError("Incorrect type of Fiuri Dense Connection")
        self.type = type
        self.n_pre, self.n_post = n_pre, n_post

        self.register_buffer("w_mask", w_mask) 
        self.w = nn.Parameter(torch.empty(n_pre, n_post))
        nn.init.kaiming_uniform_(self.w)

    def forward(self, o_pre):
        """  
        o_pre: (B, n_pre) presyn output at current step

        Returns:
            weighed presynaptic outputs
        """
        # Enforce positive weights before applying the mask
        w_pos = F.softplus(self.w)  # (n_pre, n_post)
        w_eff = w_pos * self.w_mask  # (n_pre, n_post)
        contrib = o_pre.matmul(w_eff)  # (B, n_post)
        if self.type == "EX":
            return contrib
        else:  # "IN"
            return -contrib


class FiuriSparseGJConn(nn.Module):
    """ 
        This class is only to be used to connect FIURI layers with GJ conns

        Contains a sparse graph of edges between pre and post synaptic layers.
        The GJ sign rule is applied on each target neuron, 
            thus, out GJ values are returned by forward to the next layer to be used.
        Params:
            - n_pre, n_post: pre and post synapitc layers of the connection
            - gj_edges: pre X post matrix containing non-zero values on the desired edges
    """
    def __init__(self, n_pre, n_post, gj_edges: torch.Tensor):
        super().__init__()
        self.n_pre, self.n_post = n_pre, n_post
        if gj_edges.ndim != 2 or gj_edges.shape[0] != 2:
            raise ValueError("gj_edges must be shape (2, E)")
        self.register_buffer("gj_idx", gj_edges.long())
        E = gj_edges.shape[1]
        self.gj_w = nn.Parameter(torch.empty(E))
        nn.init.normal_(self.gj_w)

    def forward(self, o_pre):
        # Return bundle
        # Return positive weights to respect GJ constraint
        w_pos = F.softplus(self.gj_w)
        return (self.gj_idx[0], self.gj_idx[1], w_pos)

# Gradient flow improvement implementation

class FIURIModuleV2(nn.Module):
    """  
    Nerual layer of the FIURI model. (fully pytorch)
    Sn = En + sum(I_jin) j=1..m where m is ammount of connections(9) 
         
    Ij in = 
            ωj * Oj if Oj ≥ En y gap junct. 
            -ωj * Oj if Oj < En y gap junct. 
            ωj * Oj chemical excitatory 
            -ωj * Oj chemical inhibitory
    where Oj is the output state of the presynaptic neuron j and ωj is the weight of the connection from neuron j to neuron n.
    
    On = ( Sn- Tn if Sn > Tn 
         ( 0 other case (10) 
    
        | Sn - Tn if Sn > Tn
    En =  En - dn if Sn ≤ Tn and Sn = En  (11) 
        \ Sn other case  
    
    where:
        - En and On represent the internal state and the output state of neuron n, respectively. (not learnable)
        - Sn represents the stimiulus comming through the connections due to the currents Ij_in.
        - Eqs 10,11 represent the dynamic of the neuron.
        - Tn and dn are the neuronal parameters that have to be learned and represent:
            - Tn the firing threshold
            - dn the decay factor (due to not enough stimulus)
    
    Each neuron has 3 channels for communication with FIURI_connections
        - 0: EX
        - 1: IN
        - 2: GJ
    """
    def __init__(
        self,
        num_cells: Optional[int] = None,
        initial_in_state: Optional[float] = 0.0,   # scalar default
        initial_out_state: Optional[float] = 0.0,  # scalar default
        initial_threshold: Optional[float] = 1.0,
        initial_decay: Optional[float] = 0.1,
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
        rnd_init: bool = False,

        # SG- parameters
        steepness_gj: float = 10.0,
        steepness_fire: float = 10.0,
        steepness_input: float = 5.0,
        input_thresh: float = 0.01,
        leaky_slope: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__()
        assert (num_cells is not None), "Must provide number of neurons per layer"      
        # save num of cells
        self.num_cells = num_cells

        self.steepness_gj = steepness_gj
        self.steepness_fire = steepness_fire
        self.steepness_input = steepness_input
        self.input_thresh = input_thresh
        self.leaky_slope = leaky_slope

        self.threshold = nn.Parameter(torch.empty(num_cells), requires_grad=True) 
        self.decay     = nn.Parameter(torch.empty(num_cells), requires_grad=True)
        
        if rnd_init:
            # TODO: Check and define proper random init ranges
            nn.init.uniform_(self.threshold, -1.0, 2.0)
            nn.init.uniform_(self.decay, 0.05, 1.0)
        else:
            nn.init.constant_(self.threshold, initial_threshold)
            nn.init.constant_(self.decay, initial_decay)
        
        self._init_E = float(initial_in_state)
        self._init_O = float(initial_out_state)

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max


    def _compute_gj_sum(self, gj_bundle, o_pre: torch.Tensor, current_in_state: torch.Tensor, out_buffer: torch.Tensor) -> None:
        """
        Computes GJ sum and scatters it IN-PLACE into out_buffer.
        """
        src, dst, w = gj_bundle
        if src.numel() == 0:
            return  # No-op, out_buffer is unchanged

        B = o_pre.size(0)
        Oj = o_pre[:, src]              # (B, E_gj)
        En = current_in_state[:, dst]   # (B, E_gj)
        
        sgn = torch.tanh(self.steepness_gj * (Oj - En)) # Tanh is most used for smooth sign
        contrib = Oj * w * sgn          # (B, E_gj)

        # out buffer modified in-place
        _scatter_add_batched(contrib, dst, out_buffer)

    def _calculate_new_state(self, S, E, T, D):
        """
        Calculates new (E, O) state from S, E, T, and D.
        """
        # leaky relu allows some gradient flow when S < T
        new_o = F.leaky_relu(S - T, negative_slope=self.leaky_slope)
        
        # we force positive D trying to enforce training stability
        D_positive = F.softplus(D) 
        
        # fire probability gate (0,1)
        fire_prob = torch.sigmoid(self.steepness_fire * (S - T))
        
        #  ~0 if S == E, y ~1 if S != E.
        diff = torch.abs(S - E)
        input_prob = torch.sigmoid(self.steepness_input * (diff - self.input_thresh))
        
        # output cases:
        E_fired = new_o          # 1: (S > T) -> E_next = O_next
        E_decay = E - D_positive # 2: (S == E) -> E_next = E - D. 
        E_subthresh = S          # 3: (S <= T y S != E) -> E_next = S.

        # E_nonfired = (if S != E: E_subthresh else: E_decay)
        E_nonfired = input_prob * E_subthresh + (1 - input_prob) * E_decay
        
        # new_e = (if S > T: E_fired else: E_nonfired)
        new_e = fire_prob * E_fired + (1 - fire_prob) * E_nonfired
        
        return new_o, (new_e, new_o)

    def neuron_step(self, state):
        """
        Performs one step of the neuron dynamics.
        If S is None, uses S = in_state (i.e., as if input_current were zero).
        """
        E, O = state if state else (self._init_E, self._init_O)
        S = torch.clamp(E, self.clamp_min, self.clamp_max)

        return self._calculate_new_state(S, E, self.threshold, self.decay)

    def forward(self, chem_influence, state = None, gj_bundle = None, o_pre = None) -> torch.Tensor:
        B = chem_influence.size(0)
        
        # Extract current state for use in gap junction computation
        E, O = state if state else (self._init_E, self._init_O)
        
        if gj_bundle is not None:
            assert o_pre is not None, "o_pre must be provided when gj_bundle is not None"
            # Pass current state to gap junction computation
            current_E_tensor = E if isinstance(E, torch.Tensor) else torch.full((B, self.num_cells), E, dtype=chem_influence.dtype, device=chem_influence.device)
            
            self._compute_gj_sum(
                gj_bundle, 
                o_pre, 
                current_in_state=current_E_tensor, 
                out_buffer=chem_influence
            )
        
        # Note: E is still the *original* state[0]
        S = torch.clamp(E + chem_influence, self.clamp_min, self.clamp_max)

        return self._calculate_new_state(S, E, self.threshold, self.decay)