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
        **kwargs,
    ) -> None:
        super().__init__()
        assert (num_cells is not None), "Must provide number of neurons per layer"      
        # save num of cells
        self.num_cells = num_cells
        # --- learnable per-neuron parameters (shape: (n,))
        t_init = torch.full((num_cells,), float(initial_threshold), dtype=torch.float32)
        d_init = torch.full((num_cells,), float(initial_decay),    dtype=torch.float32)

        self.threshold = nn.Parameter(t_init, requires_grad=True) 
        self.decay     = nn.Parameter(d_init, requires_grad=True)
        # --- persistent state buffers (allocated lazily with correct batch/device/dtype)
        
        self.register_buffer("in_state", torch.tensor(initial_in_state, dtype=torch.float32))  # E (batch, n)
        self.register_buffer("out_state", torch.tensor(initial_out_state, dtype=torch.float32))  # O/output (batch, n) — BindsNET convention      
        
        # initialize state buffers as (1, num_cells) rows filled with the initial values
        self.in_state = torch.full((1, self.num_cells), float(initial_in_state), dtype=torch.float32)
        self.out_state = torch.full((1, self.num_cells), float(initial_out_state), dtype=torch.float32)
        # defaults for initial states (rows used at first allocation)
        self._init_E = float(initial_in_state)
        self._init_O = float(initial_out_state)

        # clamp range for S_n (optional)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def _ensure_state(self, B: int, device, dtype):
        shape_mismatch = (self.in_state.ndim != 2) or (self.in_state.shape != (B, self.num_cells))
        if shape_mismatch:
            # Preserve existing state where available and only initialize new rows.
            prev_in = self.in_state if self.in_state.ndim == 2 else None
            prev_out = self.out_state if self.out_state.ndim == 2 else None

            new_in = torch.full((B, self.num_cells), self._init_E, device=device, dtype=dtype)
            new_out = torch.full((B, self.num_cells), self._init_O, device=device, dtype=dtype)

            if prev_in is not None:
                rows_to_copy = min(B, prev_in.shape[0])
                new_in[:rows_to_copy] = prev_in[:rows_to_copy].to(device=device, dtype=dtype)
            if prev_out is not None:
                rows_to_copy = min(B, prev_out.shape[0])
                new_out[:rows_to_copy] = prev_out[:rows_to_copy].to(device=device, dtype=dtype)

            self.in_state = new_in
            self.out_state = new_out
        else:
            # Ensure buffers live on the requested device/dtype without altering values.
            if self.in_state.device != device or self.in_state.dtype != dtype:
                self.in_state = self.in_state.to(device=device, dtype=dtype)
            if self.out_state.device != device or self.out_state.dtype != dtype:
                self.out_state = self.out_state.to(device=device, dtype=dtype)

    @torch.no_grad()
    def reset_state(self, B: int = None, device=None, dtype=None):
        if B is not None:
            self._ensure_state(B, device or self.in_state.device, dtype or self.in_state.dtype)
        self.in_state.fill_(self._init_E)
        self.out_state.fill_(self._init_O)
        self.in_state  = self.in_state.detach()
        self.out_state = self.out_state.detach()
    
    def detach(self):
        """  
            Detaches states tensors from graph, intended to be used while learning
        """
        self.in_state = self.in_state.detach()
        self.out_state = self.out_state.detach()

    def _compute_gj_sum(self, gj_bundle, o_pre: torch.Tensor, current_in_state: torch.Tensor = None) -> torch.Tensor:
        src, dst, w = gj_bundle
        if src.numel() == 0:
            return (current_in_state if current_in_state is not None else self.in_state).new_zeros(current_in_state.shape if current_in_state is not None else self.in_state.shape)  # (B, n)

        B = o_pre.size(0)
        Oj = o_pre[:, src]              # (B, E_gj)
        print(f"o_pre {o_pre}")
        # Use current state if provided, otherwise fall back to self.in_state
        En_state = current_in_state if current_in_state is not None else self.in_state
        En = En_state[:, dst]          # (B, E_gj)
        
        sgn = torch.where(Oj >= En, 1.0, -1.0)
        contrib = Oj * w * sgn          # (B, E_gj)
        print(f"contrib: {contrib}")

        out = En_state.new_zeros(B, self.num_cells)
        _scatter_add_batched(contrib, dst, out)
        return out  # (B, n)

    def neuron_step(self, state):
        """
        DEPRECATED
        Performs one step of the neuron dynamics.
        If S is None, uses S = in_state (i.e., as if input_current were zero).
        """
        E, O = state if state else (self._init_E, self._init_O)
        S = torch.clamp(E, self.clamp_min, self.clamp_max)

        T = self.threshold    # differentiable parameters
        D = self.decay
        
        eps = 1e-6
        eqE = (S - self.in_state).abs() <= eps

        gt = S > T
        mask = (~gt) & eqE

        new_o = F.relu(S - T)
        new_e = torch.where(S > T, new_o, torch.where(mask, self.in_state - D, S))

        return new_o, (new_e, new_o)

    def forward(self, chem_influence, state = None, gj_bundle = None, o_pre = None) -> torch.Tensor:
        B = chem_influence.size(0)
        self._ensure_state(B, chem_influence.device, chem_influence.dtype)
        
        # Extract current state for use in gap junction computation
        E, O = state if state else (self._init_E, self._init_O)
        
        if gj_bundle is not None:
            assert o_pre is not None, "o_pre must be provided when gj_bundle is not None"
            # Pass current state to gap junction computation
            current_E_tensor = E if isinstance(E, torch.Tensor) else torch.full((B, self.num_cells), E, dtype=chem_influence.dtype, device=chem_influence.device)
            gj_sum = self._compute_gj_sum(gj_bundle, o_pre, current_in_state=current_E_tensor)  # (B, n)
            chem_influence  = chem_influence + gj_sum          # (B, n)
        
        S = torch.clamp(E + chem_influence, self.clamp_min, self.clamp_max)

        T = self.threshold    # differentiable parameters
        D = self.decay
        
        eps = 1e-6
        eqE = (S - E).abs() <= eps
        #eqE = S == E 

        gt = S > T
        mask = (~gt) & eqE

        new_o = F.relu(S - T)
        new_e = torch.where(S > T, new_o, torch.where(mask, E - D, S))

        return new_o, (new_e, new_o)

    @torch.no_grad()
    def set_internal_state(
        self,
        in_s: torch.Tensor | np.ndarray,
        *,
        B: int | None = None,
        batch: int | slice | list | torch.Tensor | None = None,
        clone: bool = True,
    ):
        """
        Set the internal state E.
        Args:
          in_s:  shape (n,) or (B, n). numpy or torch.
          B:     batch size to (re)allocate if state not yet allocated. Ignored if state already (B,n) or in_s is (B,n).
          batch: select which batch rows to set when in_s is (n,). If None, broadcasts to all rows.
          clone: if True, copies data into buffers (recommended).
        """
        # Convert to tensor
        if not torch.is_tensor(in_s):
            in_s = torch.as_tensor(in_s)

        # Determine target B and allocate if needed
        current_B = self.in_state.shape[0] if self.in_state.ndim == 2 else None
        if in_s.ndim == 2:
            B_target, n = in_s.shape
            if n != self.num_cells:
                raise ValueError(f"in_s has wrong n: got {n}, expected {self.num_cells}")
            self._ensure_state(B_target, in_s.device, in_s.dtype)
            if clone:
                self.in_state.copy_(in_s)
            else:
                self.in_state = in_s
            return

        if in_s.ndim != 1 or in_s.shape[0] != self.num_cells:
            raise ValueError(f"in_s must be (n,) or (B,n). Got {tuple(in_s.shape)}; n={self.num_cells}")

        # Figure out B to use
        if current_B is None:
            B_effective = B if B is not None else 1
            self._ensure_state(B_effective, in_s.device, in_s.dtype)
        else:
            B_effective = current_B
            # ensure device/dtype
            if self.in_state.device != in_s.device or self.in_state.dtype != in_s.dtype:
                in_s = in_s.to(device=self.in_state.device, dtype=self.in_state.dtype)

        # Assign: either broadcast to all rows or selective rows
        if batch is None:
            if clone:
                self.in_state.copy_(in_s.view(1, -1).expand(B_effective, -1))
            else:
                self.in_state = in_s.view(1, -1).expand(B_effective, -1)
        else:
            self.in_state[batch] = in_s  # broadcasting to selected rows is fine

        self.in_state = self.in_state.detach()


    @torch.no_grad()
    def set_output_state(
        self,
        out_s: torch.Tensor | np.ndarray,
        *,
        B: int | None = None,
        batch: int | slice | list | torch.Tensor | None = None,
        clone: bool = True,
    ):
        """
        Set the output state O.
        Args:
          out_s: shape (n,) or (B, n). numpy or torch.
          B:     batch size to (re)allocate if state not yet allocated. Ignored if state already (B,n) or out_s is (B,n).
          batch: select which batch rows to set when out_s is (n,). If None, broadcasts to all rows.
          clone: if True, copies data into buffers (recommended).
        """
        if not torch.is_tensor(out_s):
            out_s = torch.as_tensor(out_s)

        current_B = self.out_state.shape[0] if self.out_state.ndim == 2 else None
        if out_s.ndim == 2:
            B_target, n = out_s.shape
            if n != self.num_cells:
                raise ValueError(f"out_s has wrong n: got {n}, expected {self.num_cells}")
            self._ensure_state(B_target, out_s.device, out_s.dtype)
            if clone:
                self.out_state.copy_(out_s)
            else:
                self.out_state = out_s
            return

        if out_s.ndim != 1 or out_s.shape[0] != self.num_cells:
            raise ValueError(f"out_s must be (n,) or (B,n). Got {tuple(out_s.shape)}; n={self.num_cells}")

        # Determine B
        if current_B is None:
            B_effective = B if B is not None else 1
            self._ensure_state(B_effective, out_s.device, out_s.dtype)
        else:
            B_effective = current_B
            if self.out_state.device != out_s.device or self.out_state.dtype != out_s.dtype:
                out_s = out_s.to(device=self.out_state.device, dtype=self.out_state.dtype)

        # Assign
        if batch is None:
            if clone:
                self.out_state.copy_(out_s.view(1, -1).expand(B_effective, -1))
            else:
                self.out_state = out_s.view(1, -1).expand(B_effective, -1)
        else:
            self.out_state[batch] = out_s

        self.out_state = self.out_state.detach()

    def get_state(self) -> dict:
        return {
            "in_state":self.in_state,
            "out_state": self.out_state,
            "threshold": self.threshold,
            "decay_factor": self.decay,
        }
    

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
