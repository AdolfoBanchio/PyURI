import json
import os
import torch
from torch import nn
from utils.w_builder import build_tw_matrices
from PyURI_module import FIURIModule, FiuriDenseConn, FiuriSparseGJConn
from typing import Callable
json_path = os.path.join(os.path.dirname(__file__),"TWC_fiu.json")

def create_layer(n_neurons) -> FIURIModule:
    """  
        Creates a fiuri module layer with N neurons
    """
    return FIURIModule(
        num_cells=n_neurons,
        initial_in_state=0.0,
        initial_out_state=0.0,
        initial_threshold=0.0,
        initial_decay=0.01,
        learn_threshold=True,
        learn_decay=True,
        clamp_min=-10.0,
        clamp_max=10.0,
    )

def build_twc(obs_encoder: Callable,
              action_decoder: Callable,
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


        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            One TWC step.
                Accepts:
                  - raw observation (B, 2) -> will be encoded via the function provided.

                When creating the module a proper input encoder to the 4 sensory neurons must be provided
                if you plan to pass raw observations. Decoder must accept keyword args
                n_inputs and device (compatible with utils.twc_io_wrapper.default_obs_encoder).
            """
            device = next(self.parameters()).device

            ex_in, in_in = self.obs_encoder(x, n_inputs=4, device=device)            
            in_out = self.in_layer(ex_in - in_in)
            hid_out = None
            for _ in range(3):
                # input -> hidden
                in2hid_influence = self.in2hid_IN(in_out)
                in2hid_gj_bundle = self.in2hid_GJ(in_out)

                # hidden -> hidden (one recurrent step)
                hid_out = self.hid_layer(in2hid_influence, gj_bundle=in2hid_gj_bundle, o_pre=in_out)
                hid_ex_influence = self.hid_EX(hid_out)
                hid_in_influence = self.hid_IN(hid_out)
                hid_out = self.hid_layer(hid_ex_influence + hid_in_influence)
                self.in_layer.neuron_step()
            
            # --- hidden -> output (EX only)
            hid2out_ex_influence = self.hid2out(hid_out)
            y = self.out_layer(hid2out_ex_influence)
            
            if self.log:
                self.log_monitor()
            return y
        
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
            self.in_layer.reset_state()
            self.hid_layer.reset_state() 
            self.out_layer.reset_state() 
        
        def detach(self):
            self.in_layer.detach()
            self.hid_layer.detach()
            self.out_layer.detach()

        def log_monitor(self):
            """  
            In each layer list logs a dictonary like this
            {
            "in_state":self.in_state,
            "out_state": self.out_state,
            "threshold": self.threshold,
            "decay_factor": self.decay,
            }
            """
            self.monitor["in"].append(self.in_layer.get_state())
            self.monitor["hid"].append(self.hid_layer.get_state())
            self.monitor["out"].append(self.out_layer.get_state())


    return TWC()
