import torch
from torch import nn
from typing import Optional, Iterable, Union
from bindsnet.network.nodes import Nodes

class FIURI_node(Nodes):
    """  

    Nerual node of the FIURI model.
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
    """
    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = True,
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
        # language=rst
        """
        Abstract base class constructor.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record decaying spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        """

        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=False,
            **kwargs,
        )
        # --- learnable per-neuron parameters (shape: (n,))
        T_init = torch.full((self.n,), float(initial_threshold), dtype=torch.float32)
        d_init = torch.full((self.n,), float(initial_decay),    dtype=torch.float32)

        self.threshold = nn.Parameter(T_init) if learn_threshold else T_init
        self.decay     = nn.Parameter(d_init) if learn_decay     else d_init

        # --- persistent state buffers (allocated lazily with correct batch/device/dtype)
        self.register_buffer("in_state", torch.tensor(initial_in_state, dtype=torch.float))  # E (batch, n)
        self.register_buffer("s", torch.tensor(initial_out_state, dtype=float))  # O/output (batch, n) — BindsNET convention

        # defaults for initial states (scalars stored just to use at first allocation)
        self._init_E = float(initial_in_state)
        self._init_O = float(initial_out_state)

        # clamp range for S_n (optional)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def _compute_decays(self) -> None:
        pass
        
    def set_batch_size(self, batch_size: int) -> None:
        super().set_batch_size(batch_size)   # keeps BindsNET’s s/x/summed in sync
        device = self.threshold.device       # or a stored self._device
        dtype  = self.threshold.dtype

        self._ensure_state(batch_size, device, dtype)

    def _ensure_state(self, batch_size: int, device, dtype) -> None:
        """
        Make sure internal buffers match the current batch shape/device/dtype.
        """
        full_shape = (batch_size, *self.shape)

        if self.sum_input:
            needs_resize = self.summed.numel() == 0 or self.summed.shape != full_shape
            if needs_resize:
                self.summed = torch.zeros(full_shape, device=device, dtype=dtype)
            else:
                self.summed = self.summed.to(device=device, dtype=dtype)

        if self.in_state.numel() == 0 or self.in_state.shape != full_shape:
            self.in_state = torch.full(full_shape, self._init_E, device=device, dtype=dtype)
        else:
            self.in_state = self.in_state.to(device=device, dtype=dtype)

        if self.s.numel() == 0 or self.s.shape != full_shape:
            self.s = torch.full(full_shape, self._init_O, device=device, dtype=dtype)
        else:
            self.s = self.s.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> None:
        """
        One simulation step.
        x: presynaptic input CURRENT to this layer at this step.
        """
        batch = x.size(0) if x.dim() > 0 else 1
        device = x.device
        dtype = x.float().dtype

        # Ensure buffers are ready before letting the parent accumulate inputs.
        self._ensure_state(batch, device, dtype)

        # Keep base behavior (traces and optional input pre-sum)
        super().forward(x)

        # Get the input current for this step:
        # Previous stage manipulates the presynaptic conncetions
        # x "arrives" with all the currents summed.
        input_current = x

        # Ensure state tensors match batch/device/dtype (covers sum_input=False case)
        self._ensure_state(batch, input_current.device, input_current.dtype)

        # Broadcast learnable params to (batch, n)
        # to match input_current shape for elementwise ops
        # using .to() to ensure correct device/dtype
        # getting the treshold and decay values from this step
        T = self.threshold.to(input_current).unsqueeze(0).expand(batch, -1)
        d = self.decay.to(input_current).unsqueeze(0).expand(batch, -1)

        # --- S_n = E_n + sum(I_jn)
        S = self.in_state + input_current

        # Copying from Ariel code: TODO: ask about this
        # Optional clamp (keeps numerics in check; remove if not desired)
        S = torch.clamp(S, min=self.clamp_min, max=self.clamp_max)

        # Conditions
        gt_thresh = S > T
        # Avoid exact float equality; use tolerance if you truly need equality.
        eq_S_E    = torch.isclose(S, self.in_state, rtol=1e-6, atol=1e-6)
        le_thresh = ~gt_thresh

        # Case 1: S > T  -->  O = S - T;  E = S - T
        O_case1 = S - T
        E_case1 = S - T

        # Case 2: S <= T and S == E  -->  O = 0;  E = E - d
        O_case2 = torch.zeros_like(S)
        E_case2 = self.in_state - d

        # Case 3: otherwise          -->  O = 0;  E = S
        O_case3 = torch.zeros_like(S)
        E_case3 = S

        # Build masks
        mask1 = gt_thresh
        mask2 = le_thresh & eq_S_E
        # Select per case
        new_O = torch.where(mask1, O_case1, torch.where(mask2, O_case2, O_case3))
        new_E = torch.where(mask1, E_case1, torch.where(mask2, E_case2, E_case3))

        # Assign
        self.s        = new_O
        self.in_state = new_E

        # If we used the pre-sum buffer, zero it for next step
        if self.sum_input and (self.summed is not None):
            self.summed.zero_()

    def reset_state_variables(self) -> None:
        # Reset neuron states at episode boundaries
        if self.in_state is not None:
            self.in_state.zero_()
        if self.s is not None:
            self.s.zero_()
        # Keep traces reset from parent if enabled
        super().reset_state_variables()
        # If using summed input, zero it too
        if self.sum_input and (self.summed is not None):
            self.summed.zero_()
        
