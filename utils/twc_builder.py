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
        initial_threshold=-5.0,
        initial_decay=0.0,
        learn_threshold=True,
        learn_decay=True,
        clamp_min=-10.0,
        clamp_max=10.0,
    )

def build_twc(action_decoder: Callable,
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
            self.decoder = action_decoder

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
                Accepts either:
                  - raw observation (B, 2) -> will be encoded via decoder into (B, n_inputs, 3)
                  - already-encoded input (B, n_inputs, 3)

                When creating the module a proper input decoder to the 4 sensory neurons must be provided
                if you plan to pass raw observations. Decoder must accept keyword args
                n_inputs and device (compatible with utils.twc_io_wrapper.default_obs_encoder).
            """
            device = next(self.parameters()).device
            if x.ndim == 3: # looks like a already encoded observation
                z = x
            else: # let the encoder manage it completly
                if self.decoder is None:
                    raise ValueError("Decoder is None but raw observations were provided to TWC.forward")
                z = self.decoder(x, n_inputs=4, device=device)
            
            # Normalize encoder output layout -> ex_in, in_in (B, n_in)
            if z.ndim != 3:
                raise ValueError(f"Encoded input must be 3D; got {z.ndim}D")
            Bz, A, Bdim = z.shape
            if A == self.in_layer.num_cells and Bdim in (2, 3):
                # (B, n_in, C)
                ex_in = z[:, :, 0]
                in_in = z[:, :, 1]
            elif Bdim == self.in_layer.num_cells and A in (2, 3):
                # (B, C, n_in)
                ex_in = z[:, 0, :]
                in_in = z[:, 1, :]
            else:
                raise ValueError(f"Encoded input has incompatible shape {tuple(z.shape)} for n_in={self.in_layer.num_cells}")

            B = ex_in.size(0)
            
            in_o = self.in_layer(ex_in - in_in)

            # input -> hidden
            in_influence = self.in2hid_IN(in_o)
            in_gj_bundle = self.in2hid_GJ(in_o)

            h_o = self.hid_layer(in_influence, gj_bundle=in_gj_bundle, o_pre=in_o)

            # hidden -> hidden (one recurrent step)
            ex_h = self.hid_EX(h_o)                      # (B, n_hid)
            in_h = self.hid_IN(h_o)                      # (B, n_hid)
            h_o_2 = self.hid_layer(ex_h + in_h)          # chem_influence only

            # --- hidden -> output (EX only)
            ex_o = self.hid2out(h_o_2)                   # (B, n_out)
            y = self.out_layer(ex_o)                     # output layer step

            if self.log:
                self.log_monitor()
            return y
        
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
