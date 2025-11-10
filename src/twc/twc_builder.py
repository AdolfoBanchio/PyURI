import json
import os
import math
from typing import Callable
import torch
from torch import nn
from fiuri import FIURIModule, FiuriDenseConn, FiuriSparseGJConn
from .w_builder import build_tw_matrices

json_path = os.path.join(os.path.dirname(__file__), "TWC_fiu.json")

def build_twc(obs_encoder: Callable,
              action_decoder: Callable,
              internal_steps: int,
              log_stats: bool = True) -> nn.Module:
    """ Extracts the data from thr TWC description
    and returns a nn.Module with the TWC implementation
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(json_path, "r") as f:
        net_data = json.load(f)

    masks, sizes = build_tw_matrices(net_data)

    n_in, n_hid, n_out = sizes["n_in"], sizes["n_hid"], sizes["n_out"]

    in_layer =  FIURIModule(
        num_cells=n_in,
        initial_in_state=0.0,
        initial_out_state=0.0,
        initial_threshold=0.0,
        initial_decay=0.1,
        clamp_min=-10.0,
        clamp_max=10.0,
    )
    hid_layer = FIURIModule(
        num_cells=n_hid,
        initial_in_state=0.0,
        initial_out_state=0.0,
        initial_threshold=0.0,
        initial_decay=0.1,
        clamp_min=-10.0,
        clamp_max=10.0,
    )
    out_layer = FIURIModule(
        num_cells=n_out,
        initial_in_state=0.0,
        initial_out_state=0.0,
        initial_threshold=0.0,
        initial_decay=0.1,
        clamp_min=-10.0,
        clamp_max=10.0,
    )

    in2hid = FiuriDenseConn(n_pre=n_in, n_post=n_hid,w_mask=masks["in2hid"]["IN"], type="IN")
    hid_IN = FiuriDenseConn(n_pre=n_hid, n_post=n_hid, w_mask=masks["hid"]["IN"], type="IN")
    hid_EX = FiuriDenseConn(n_pre=n_hid, n_post=n_hid, w_mask=masks["hid"]["EX"], type="EX")
    hid2out_EX = FiuriDenseConn(n_pre=n_hid, n_post=n_out, w_mask=masks["hid2out"]["EX"], type="EX")

    # create the only GJ sparse conn
    # PLM -> PVC, AVM -> AVD
    gj_edges = torch.tensor([[1, 2],   # src (PLM=1, AVM=2)
                             [2, 1]])  # dst (PVC=2, AVD=1)
    gj_conn = FiuriSparseGJConn(n_pre=n_in, n_post=n_hid, gj_edges=gj_edges)

    
    class TWC (nn.Module):
        """  
        When creating the module a proper input decoder to the 4 sensory neurons must be provided
        if you plan to pass raw observations. Decoder must accept keyword args
        n_inputs and device (compatible with utils.twc_io_wrapper.default_obs_encoder).
     
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # neuron layers
            self.in_layer = in_layer
            self.hid_layer = hid_layer
            self.out_layer = out_layer

            # connections
            self.in2hid_IN = in2hid
            self.in2hid_GJ = gj_conn

            self.hid_IN = hid_IN
            self.hid_EX = hid_EX

            self.hid2out = hid2out_EX

            # I/O
            self.obs_encoder = obs_encoder
            self.action_decoder = action_decoder

            # MONITOR
            self.log = log_stats
            self.monitor = {
                "in": [],
                "hid": [],
                "out": [],
            }
            self._state = None
            self.internal_steps = internal_steps

        def _init_layer_state(self, layer: FIURIModule, batch_size: int, device, dtype):
            E0 = torch.full((batch_size, layer.num_cells), layer._init_E, device=device, dtype=dtype)
            O0 = torch.full((batch_size, layer.num_cells), layer._init_O, device=device, dtype=dtype)
            return (E0, O0)

        def _make_state(self, batch_size: int, device, dtype):
            return {
                "in": self._init_layer_state(self.in_layer, batch_size, device, dtype),
                "hid": self._init_layer_state(self.hid_layer, batch_size, device, dtype),
                "out": self._init_layer_state(self.out_layer, batch_size, device, dtype),
            }

        def _ensure_state(self, batch_size: int, device, dtype):
            if self._state is None:
                self._state = self._make_state(batch_size, device, dtype)
                return self._state

            sample_E, _ = self._state["in"]
            if sample_E.shape[0] != batch_size or sample_E.device != device or sample_E.dtype != dtype:
                self._state = self._make_state(batch_size, device, dtype)
            return self._state

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            One TWC step.
                Accepts:
                  - raw observation (B, 2) -> will be encoded via the function provided.

                When creating the module a proper input encoder to the 4 sensory neurons must be provided
                Assumes raw observations. Decoder must accept keyword args
                n_inputs and device (compatible with utils.twc_io_wrapper.default_obs_encoder).

                returns:
                  - REV and FWD outputs decoded by the action decoder.
            """
            device = next(self.parameters()).device
            if x.device != device:
                x = x.to(device)
            ex_in, in_in = self.obs_encoder(x, n_inputs=4, device=device)            
            B = ex_in.size(0)
            dtype = ex_in.dtype
            current_state = self._ensure_state(B, device, dtype)
            
            state = {}
            # For input layer: directly set states from encoder values (matching ariel feedNN behavior)
            # We do the same: E = encoded_value, O = encoded_value
            # 
            # The encoder returns (ex_in, in_in) where:
            #   ex_in: [PVD_EX=0, PLM_EX, AVM_EX=0, ALM_EX] - excitatory channel
            #   in_in: [PVD_IN, PLM_IN=0, AVM_IN, ALM_IN=0] - inhibitory channel
            # Input neuron order: [PVD, PLM, AVM, ALM]
            #
            # Note: Each neuron gets its value from either ex_in or in_in (the other is zero)
            # Combining them gives: [PVD_IN, PLM_EX, AVM_IN, ALM_EX]            
            # Combine ex_in and in_in to get final values for each neuron

            input_values = ex_in + in_in  # (B, 4) - combines EX and IN channels
            in_E = input_values  # Internal state = encoded value
            in_O = input_values  # Output state = encoded value (matching ariel)
            in_state = (in_E, in_O)
            
            # Ariel's two-phase update:
            # Phase 1: computeVnext() for all neurons
            #   - For input neurons: feedNN() just set states, so getOutputState() returns NEW states
            #   - For other neurons: getOutputState() returns OLD states (before commit)
            # Phase 2: commitComputation() for all neurons (commits buffered states)
            #
            # For input layer: feedNN() sets states, then computeVnext() runs (with no influence)
            in_out_new, in_state_new = self.in_layer.neuron_step(in_state)
            
            # After feedNN(), ariel uses the NEW output states for connections
            # (getOutputState() returns the state just set by feedNN)
            # So we use in_O (the newly set output state) for connections
            in_out_for_connections = in_O
            
            # Now commit: update state
            state["in"] = in_state_new

            hid_state_cur = current_state["hid"]
            hid_state_old = (hid_state_cur[0].clone(), hid_state_cur[1].clone())
            
            for _ in range(self.internal_steps):
                # Ariel's two-phase update:
                # Phase 1: computeVnext() for all neurons
                #   - Input neurons: use NEW output states (just set by feedNN)
                #   - Other neurons: use OLD output states (before commit)
                # Phase 2: commitComputation() commits new states
                in2hid_influence = self.in2hid_IN(in_out_for_connections)
                in2hid_gj_bundle = self.in2hid_GJ(in_out_for_connections)
                
                
                # For hidden layer self-connections, use OLD hidden output state
                # (matching ariel's two-phase: all connections use pre-update states)
                hid_out_old = current_state["hid"][1] if current_state["hid"] is not None else hid_out_new
                hid_ex_influence = self.hid_EX(hid_out_old)
                hid_in_influence = self.hid_IN(hid_out_old)
                hid2hid_influence = hid_ex_influence + hid_in_influence
                # Compute hidden layer new state from input connections
                hid_out_new, hid_state_new = self.hid_layer(
                    in2hid_influence + hid2hid_influence,
                    state=hid_state_old,
                    gj_bundle=in2hid_gj_bundle,
                    o_pre=in_out_for_connections,  # Use NEW input output state (after feedNN)
                )
                state["hid"] = hid_state_new
            
            # --- hidden -> output (EX only)
            # Use OLD hidden output state for connections (matching ariel's two-phase update)
            hid_out_old_final = current_state["hid"][1] if current_state["hid"] is not None else hid_out_new
            #print(f"DEBUG hid out old final {hid_out_old_final}")
            hid2out_ex_influence = self.hid2out(hid_out_old_final)
            
            out_state_cur = current_state["out"]
            out_output, out_layer_state = self.out_layer(
                hid2out_ex_influence,
                state=(out_state_cur[0].clone(), out_state_cur[1].clone())
            )
            state["out"] = out_layer_state
            self._state = state
            
            if self.log:
                self.log_monitor(state)
            
            # Decoder needs internal states (E), not output states (O)
            # Matching ariel's getFeedBackNN() which uses getInternalState()
            out_internal_states = out_layer_state[0]  # (B, 2) - internal states [REV, FWD]
            return self.action_decoder(out_internal_states)
        
        def get_action(self, y):
            """  
            Uses the action decoder provided to generate an enviroment action.
            Uses reads net output as ['REV', 'FWD']
            """
            return self.action_decoder(y)
        
        def reset(self):
            """  
                Resets the state variables for each layer
            """
            self._state = None
        
        def reset_internal_only(self):
            """
            Reset only internal states (E) to their initial value while preserving
            output states (O), to mirror Ariel's Reset behavior.
            """
            if self._state is None:
                return
            # For each layer, set E := init_E and keep O unchanged
            for layer_key, layer in (("in", self.in_layer), ("hid", self.hid_layer), ("out", self.out_layer)):
                E, O = self._state[layer_key]
                new_E = torch.full_like(E, layer._init_E)
                self._state[layer_key] = (new_E, O)
        
        def detach(self):
            if self._state is not None:
                self._state = {
                    name: (state_pair[0].detach(), state_pair[1].detach())
                    for name, state_pair in self._state.items()
                }

        def log_monitor(self, state):
            """  
            In each layer list logs a dictonary like this
            {
            "in_state":self.in_state,
            "out_state": self.out_state,
            "threshold": self.threshold,
            "decay_factor": self.decay,
            }
            """
            def _pack(layer, state_pair):
                return {
                    "in_state": state_pair[0].detach().cpu(),
                    "out_state": state_pair[1].detach().cpu(),
                    "threshold": layer.threshold.detach().cpu(),
                    "decay_factor": layer.decay.detach().cpu(),
                }

            self.monitor["in"].append(_pack(self.in_layer, state["in"]))
            self.monitor["hid"].append(_pack(self.hid_layer, state["hid"]))
            self.monitor["out"].append(_pack(self.out_layer, state["out"]))

        def _set_all_weights_to_one(self):
            """Force every learnable connection weight to produce 1 after softplus."""
            softplus_inv_one = math.log(math.e - 1.0)
            with torch.no_grad():
                dense_conns = (
                    self.in2hid_IN,
                    self.hid_IN,
                    self.hid_EX,
                    self.hid2out,
                )
                for conn in dense_conns:
                    conn.w.fill_(softplus_inv_one)
                self.in2hid_GJ.gj_w.fill_(softplus_inv_one)
        

    return TWC()
