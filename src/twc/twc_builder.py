import json
import os
from typing import Callable

import torch
from torch import nn

from fiuri import FIURIModule, FiuriDenseConn, FiuriSparseGJConn
from .w_builder import build_tw_matrices

json_path = os.path.join(os.path.dirname(__file__), "TWC_fiu.json")

def create_layer(n_neurons) -> FIURIModule:
    """  
        Creates a fiuri module layer with N neurons
    """
    return FIURIModule(
        num_cells=n_neurons,
        initial_in_state=0.0,
        initial_out_state=0.0,
        initial_threshold=0.0,
        initial_decay=0.1,
        clamp_min=-10.0,
        clamp_max=10.0,
    )

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

    in_layer =  create_layer(n_in)
    hid_layer = create_layer(n_hid)
    out_layer = create_layer(n_out)

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
            # Clone states to avoid in-place modifications on tensors needed for autograd
            in_state_cur = current_state["in"]
            in_state = (in_state_cur[0].clone(), in_state_cur[1].clone())
            in_out, new_in_state = self.in_layer(ex_in + in_in, state=in_state)
            state["in"] = new_in_state

            hid_state_cur = current_state["hid"]
            hid_state = (hid_state_cur[0].clone(), hid_state_cur[1].clone())
            hid_out = None
            
            for _ in range(self.internal_steps):
                in2hid_influence = self.in2hid_IN(in_out)
                in2hid_gj_bundle = self.in2hid_GJ(in_out)
                
                hid_out, hid_state = self.hid_layer(
                    in2hid_influence,
                    state=hid_state,
                    gj_bundle=in2hid_gj_bundle,
                    o_pre=in_out,
                )
                hid_ex_influence = self.hid_EX(hid_out)
                hid_in_influence = self.hid_IN(hid_out)
                hid_out, hid_state = self.hid_layer(
                    hid_ex_influence + hid_in_influence,
                    state=hid_state,
                )
                state["hid"] = hid_state
                # just perform an internal step without external influence
                #in_out, new_in_state = self.in_layer.neuron_step(state=state["in"]) 
                #state["in"] = new_in_state

            
            # --- hidden -> output (EX only)
            hid2out_ex_influence = self.hid2out(hid_out)
            out_state_cur = current_state["out"]
            out_state, out_layer_state = self.out_layer(
                hid2out_ex_influence,
                state=(out_state_cur[0].clone(), out_state_cur[1].clone())
            )
            state["out"] = out_layer_state
            self._state = state
            
            if self.log:
                self.log_monitor(state)
            return self.action_decoder(out_state)
        
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


    return TWC()
