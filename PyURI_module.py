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
    
    On = ( Sn- Tn if Sn > Tn 
         ( 0 other case (10) 
    
        | Sn - Tn if Sn > Tn
    En =  En - dn if Sn ≤ Tn and Sn = En  (11) 
        \ Sn other case  
         
    Ij in = 
            ωj * Oj if Oj ≥ En y gap junct. 
            -ωj * Oj if Oj < En y gap junct. 
            ωj * Oj chemical excitatory 
            -ωj * Oj chemical inhibitory
    where Oj is the output state of the presynaptic neuron j and ωj is the weight of the connection from neuron j to neuron n.
    
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
        learn_threshold: bool = True,
        learn_decay: bool = True,
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

        self.threshold = nn.Parameter(t_init, requires_grad=True) if learn_threshold else t_init
        self.decay     = nn.Parameter(d_init, requires_grad=True) if learn_decay     else d_init

        # --- persistent state buffers (allocated lazily with correct batch/device/dtype)
        """
        self.register_buffer("in_state", torch.tensor(initial_in_state, dtype=torch.float32))  # E (batch, n)
        self.register_buffer("out_state", torch.tensor(initial_out_state, dtype=torch.float32))  # O/output (batch, n) — BindsNET convention      
        """
        self.in_state = torch.tensor(initial_in_state, dtype=torch.float32)
        self.out_state = torch.tensor(initial_out_state, dtype=torch.float32)
        # defaults for initial states (scalars stored just to use at first allocation)
        self._init_E = float(initial_in_state)
        self._init_O = float(initial_out_state)

        # clamp range for S_n (optional)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    # Functions to use Thr and decay as positive values during evaluation and training
    def thr_pos(self):
        return F.softplus(self.threshold)
    
    def dec_pos(self):
        return F.softplus(self.decay)

    def _ensure_state(self, B: int, device, dtype):
        if self.in_state.ndim != 2 or self.in_state.shape != (B, self.num_cells):
            self.in_state  = torch.full((B, self.num_cells), self._init_E, device=device, dtype=dtype)
            self.out_state = torch.full((B, self.num_cells), self._init_O, device=device, dtype=dtype)

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

    def _compute_gj_sum(self, gj_bundle, o_pre: torch.Tensor) -> torch.Tensor:
        src, dst, w = gj_bundle
        if src.numel() == 0:
            return self.in_state.new_zeros(self.in_state.shape)  # (B, n)

        B = o_pre.size(0)
        Oj = o_pre[:, src]              # (B, E_gj)
        En = self.in_state[:, dst]      # (B, E_gj)
        sgn = torch.where(Oj >= En, 1.0, -1.0)
        contrib = Oj * w * sgn          # (B, E_gj)

        out = self.in_state.new_zeros(B, self.num_cells)
        _scatter_add_batched(contrib, dst, out)
        return out  # (B, n)

    def neuron_step(self, S=None):
        """
        Performs one step of the neuron dynamics.
        If S is None, uses S = in_state (i.e., as if input_current were zero).
        """
        if S is None:
            S = self.in_state

        # Make sure params are on the same device/dtype as S
        T = self.thr_pos().view(1, -1)
        d = self.dec_pos().view(1, -1)

        eps_abs = 1e-8
        eps_rel = 1e-6
        no_stim = (S.abs() <= (eps_abs + eps_rel * (self.in_state.abs() + 1.0)))

        gt = S > T
        mask = (~gt) & no_stim
        new_o = F.relu(S - T)
        new_e = torch.where(S > T, new_o, torch.where(mask, self.in_state - d, S))

        self.out_state = new_o
        self.in_state  = new_e

    def forward(self, ex_raw, in_raw, gj_bundle, o_pre) -> torch.Tensor:
        B = ex_raw.size(0)
        self._ensure_state(B, ex_raw.device, ex_raw.dtype)
        
        chem_influence = ex_raw - in_raw                  # (B, n)
        gj_sum         = self._compute_gj_sum(gj_bundle, o_pre)  # (B, n)
        input_current  = chem_influence + gj_sum          # (B, n)

        S = torch.clamp(self.in_state + input_current, self.clamp_min, self.clamp_max)
        self.neuron_step(S=S)
        return self.out_state

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



class FiuriSparseConn(nn.Module):
    """ 
        This class is only to be used to connect FIURI layers in order 
        to represent sparse connections between neurons.

        Contains a sparse graph of edges between pre and post synaptic layers.
        Used for arbitrary wiring between layers.
        The GJ sign rule is applied on each target neuron, 
            thus, out GJ values are returned by forward to the next layer to be used.
        Params:
            - n_re, n_post: pre and post synapitc layers of the connection
            - ex_edges: pre X post matrix containing non-zero values on the desired edges
            - in_edges: pre X post matrix containing non-zero values on the desired edges
            - gj_edges: pre X post matrix containing non-zero values on the desired edges
    """

    def __init__(self,n_pre,n_post,
                 ex_edges,in_edges,gj_edges,
                 ex_init=0.1, in_init=0.1, gj_init=0.1):
        """ex_edges/in_edges/gj_edges: LongTensor edge_index of shape (2, E)
        where row 0 = src indices in [0, n_pre), row 1 = dst indices in [0, n_post)
        """
        super().__init__()
        self.n_pre, self.n_post = n_pre, n_post
        # Register edge indices as buffers (not learnable)
        self.register_buffer("ex_idx", ex_edges)   # (2, E_ex)
        self.register_buffer("in_idx", in_edges)   # (2, E_in)
        self.register_buffer("gj_idx", gj_edges)   # (2, E_gj)
        # Per-edge learnable weights
        self.ex_w = nn.Parameter(torch.empty(self.ex_idx.shape[1]).normal_(0, ex_init), requires_grad=True)
        self.in_w = nn.Parameter(torch.empty(self.in_idx.shape[1]).normal_(0, in_init), requires_grad=True)
        self.gj_w = nn.Parameter(torch.empty(self.gj_idx.shape[1]).normal_(0, gj_init), requires_grad=True)

    def forward(self, o_pre):
        """
        o_pre: (B, n_pre) presyn output at current step.
        Returns:
            ex_raw: (B, n_post) ==  w_ex * O_j
            in_raw: (B, n_post) ==  w_in * O_j
            gj_bundle: tuple used to extract GJ sign downstream:
                (src_gj, dst_gj, w_gj) as tensors.

        summation of each channel is handled in the post synaptic layer
        """
        B = o_pre.size(0)
        device = o_pre.device
        dtype  = o_pre.dtype

        ex_raw = o_pre.new_zeros(B, self.n_post)
        in_raw = o_pre.new_zeros(B, self.n_post)

        # --- EX 
        if self.ex_idx.numel() > 0:
            src, dst = self.ex_idx[0], self.ex_idx[1]   # (E_ex,)
            Oj = o_pre[:, src]                          # (B, E_ex)
            contrib = Oj * self.ex_w                    # (B, E_ex)
            ex_raw.zero_()
            _scatter_add_batched(contrib, dst, ex_raw)

        # --- IN 
        if self.in_idx.numel() > 0:
            src, dst = self.in_idx[0], self.in_idx[1]    # (E_in,)
            Oj = o_pre[:, src]                           # (B, E_in)
            contrib = Oj * self.in_w                     # (B, E_in)
            in_raw.zero_()
            _scatter_add_batched(contrib, dst, in_raw)

        # --- GJ bundle for exact sign in postsyn layer
        gj_bundle = (self.gj_idx[0], self.gj_idx[1], self.gj_w)

        return ex_raw, in_raw, gj_bundle
