import torch
import numpy as np
from torch import nn
from typing import Optional, Sequence, Tuple, Union
from bindsnet.network.nodes import Nodes, CSRMNodes
from bindsnet.network.topology import AbstractConnection

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
        debug: Optional[bool] = False,
        num_cells: Optional[int] = None,
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
            shape=(num_cells,3),
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
            **kwargs,
        )
        self.debug = debug
        # save num of cells
        self.num_cells = num_cells
        # --- learnable per-neuron parameters (shape: (n,))
        T_init = torch.full((num_cells,), float(initial_threshold), dtype=torch.float32)
        d_init = torch.full((num_cells,), float(initial_decay),    dtype=torch.float32)

        self.threshold = nn.Parameter(T_init) if learn_threshold else T_init
        self.decay     = nn.Parameter(d_init) if learn_decay     else d_init

        # --- persistent state buffers (allocated lazily with correct batch/device/dtype)
        self.register_buffer("in_state", torch.tensor(initial_in_state, dtype=torch.float))  # E (batch, n)
        self.register_buffer("out_state", torch.tensor(initial_out_state, dtype=torch.float))  # O/output (batch, n) — BindsNET convention
        self.s = torch.tensor(torch.tensor(initial_out_state, dtype=torch.float))
        # defaults for initial states (scalars stored just to use at first allocation)
        self._init_E = float(initial_in_state)
        self._init_O = float(initial_out_state)

        # clamp range for S_n (optional)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def _compute_decays(self) -> None:
        pass

    def set_batch_size(self, batch_size) -> None:
        super().set_batch_size(batch_size)
        
        dev, dt = self.threshold.device, self.threshold.dtype
        shape = (batch_size, self.num_cells)
        if self.in_state is None or self.in_state.shape != shape:
            self.in_state = torch.full(shape, self._init_E, device=dev, dtype=dt)
        if self.out_state is None or self.out_state.shape != shape:
            self.out_state = torch.full(shape, self._init_O, device=dev, dtype=dt)
        
    
    def forward(self, x: torch.Tensor) -> None:
        """
        One simulation step.
        x: presynaptic input CURRENT to this layer at this step.
        We expect x to be -> (batch, n, 3) where:
                    - x[..., 0] = O_exc, x[..., 1] = O_inh, x[..., 2] = O_gj
                    - batch: numbers of samples procesed in parallel.
        """
        batch = x.shape[0]
        # Keep base behavior (traces), sum_input always False, because we are using 3 channels
        super().forward(x)
        
        # Broadcast learnable params to (batch, n)
        # to match input_current shape for elementwise ops
        # using .to() to ensure correct device/dtype
        # getting the treshold and decay values from this step
        T = self.threshold.view(1, -1).to(self.in_state) # (1, num_cells) -> (batch, num_cells)
        d = self.decay.view(1, -1).to(self.in_state) # (batch, n)
        
        assert T.shape[1] == self.num_cells and d.shape[1] == self.num_cells

        exc = self.summed[..., 0] # (batch, n)
        inh = self.summed[..., 1] # (batch, n)
        gj  = self.summed[..., 2] # (batch, n)

        for name, t in [('exc', exc), ('inh', inh), ('gj', gj)]:
            assert t.shape == (batch, self.num_cells), (name, t.shape)

        # chemical influcene +exc - inh 
        # (assumes the input arrives with all the corresponing inputs summed and weighted)
        chem_influence = +exc - inh

        # TODO: verify if equality must be checked ASK!
        # V1: Ignores equallity between the input of the Gj and the Internal state
        gj_sign = torch.where(gj > self.in_state, 1.0, 
                            torch.where(gj < self.in_state, -1.0, 0.0))
        
        # V2: takes into the account the equality
        #gj_sign = torch.where(gj >= self.in_state, 1.0, -1.0)
        
        gj_curr = gj_sign * gj

        input_current = chem_influence + gj_curr 

        # Compute the stimulus coming trhough the neuron
        # --- S_n = E_n + sum(I_jn)
        S = self.in_state + input_current
        # Copying from Ariel code: TODO: ask about this
        # clamp (keeps numerics in check; remove if not desired)
        S = torch.clamp(S, min=self.clamp_min, max=self.clamp_max)

        if self.debug:
            print(f'inh inf:{inh}')
            print(f'exc inf:{exc}')
            print(f'gj inf inf:{gj_curr}')
            print('current influence (sum Ij):', input_current)
            print('Stimulus (instate + Ij)', S)
        # Conditions
        gt_thresh = S > T
        # Avoid exact float equality; use tolerance if you truly need equality.
        eq_S_E    = S == self.in_state
        le_thresh = ~gt_thresh

        # Case 1: S > T  -->  O = S - T;  E = S - T
        O_c1 = S - T
        E_c1 = S - T

        # Case 2: S <= T and S == E  -->  O = 0;  E = E - d
        O_c2 = torch.zeros_like(S)
        E_c2 = self.in_state - d

        # Case 3: otherwise          -->  O = 0;  E = S
        O_c3 = torch.zeros_like(S)
        E_c3 = S

        # Build masks
        mask1 = gt_thresh
        mask2 = le_thresh & eq_S_E

        # Select per case
        new_O = torch.where(mask1, O_c1, torch.where(mask2, O_c2, O_c3))
        new_E = torch.where(mask1, E_c1, torch.where(mask2, E_c2, E_c3))

        # Assign
        self.out_state = new_O
        self.in_state  = new_E

        if self.debug:
            print(f'Final out_state: {self.out_state}')
            print(f'Final in_state: {self.in_state}')
            print('==== End of update ====')
        # If we used the pre-sum buffer, zero it for next step
        if self.sum_input and (self.summed is not None):
            self.summed.zero_()
        
        # Expose activity with 3 channels for downstream connections:
        # channel 0 carries the neuron's output; channels 1-2 are zero.
        # Shape: (batch, num_cells, 3) matching layer.shape for flattening.
        zeros = torch.zeros_like(self.out_state)
        self.s = torch.stack([self.out_state, zeros, zeros], dim=-1)
        return self.out_state

    def reset_state_variables(self) -> None:
        """ 
        Reset neuron states at episode boundaries
        TODO: see what actually needs to be reset, because in 
                and out state are important for future forward computations 
        """
        super().reset_state_variables()
        

class FIURI_Connection(AbstractConnection):
    # language=rst
    """
    Specifies synapses between one or two populations of neurons.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a :code:`Connection` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
         :param nu: Learning rate for both pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param Union[float, torch.Tensor] wmin: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param Union[float, torch.Tensor] wmax: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        w = kwargs.get("w", None)
        if w is None:
            if (self.wmin == -np.inf).any() or (self.wmax == np.inf).any():
                w = torch.clamp(torch.rand(source.n, target.n), self.wmin, self.wmax)
            else:
                w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin)
        else:
            if (self.wmin != -np.inf).any() or (self.wmax != np.inf).any():
                w = torch.clamp(torch.as_tensor(w), self.wmin, self.wmax)

        self.w = nn.Parameter(w, requires_grad=True)

        b = kwargs.get("b", None)
        if b is not None:
            self.b = nn.Parameter(b, requires_grad=True)
        else:
            self.b = None

        if isinstance(self.target, CSRMNodes):
            self.s_w = None

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using connection weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
                 decaying spike activation).
        """
        # Compute multiplication of spike activations by weights and add bias.
        if self.b is None:
            post = s.view(s.size(0), -1).float() @ self.w
        else:
            post = s.view(s.size(0), -1).float() @ self.w + self.b
        return post.view(s.size(0), *self.target.shape)

    def compute_window(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """ """

        if self.s_w == None:
            # Construct a matrix of shape batch size * window size * dimension of layer
            self.s_w = torch.zeros(
                self.target.batch_size, self.target.res_window_size, *self.source.shape
            )

        # Add the spike vector into the first in first out matrix of windowed (res) spike trains
        self.s_w = torch.cat((self.s_w[:, 1:, :], s[:, None, :]), 1)

        # Compute multiplication of spike activations by weights and add bias.
        if self.b is None:
            post = (
                self.s_w.view(self.s_w.size(0), self.s_w.size(1), -1).float() @ self.w
            )
        else:
            post = (
                self.s_w.view(self.s_w.size(0), self.s_w.size(1), -1).float() @ self.w
                + self.b
            )

        return post.view(
            self.s_w.size(0), self.target.res_window_size, *self.target.shape
        )

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        pass
    
    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            w_abs_sum = self.w.abs().sum(0).unsqueeze(0)
            w_abs_sum[w_abs_sum == 0] = 1.0
            self.w *= self.norm / w_abs_sum

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()
