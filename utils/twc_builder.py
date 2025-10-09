import json
import os
import torch
from torch import nn
from utils.w_builder import build_tw_edges
from PyURI_module import FIURIModule, FiuriSparseConn
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
        initial_threshold=-10.0,
        initial_decay=0.1,
        learn_threshold=True,
        learn_decay=True,
        clamp_min=-10.0,
        clamp_max=10.0,
    )

def make_sparse_modules_from_tw_edges(edges_dict, device=None):
    """
    Builds three FiuriUnsignSparseConn modules from build_tw_edges() output.
    Returns (in2hid, hid, hid2out).
    """
    sizes = edges_dict["sizes"]
    n_in, n_hid, n_out = sizes["n_in"], sizes["n_hid"], sizes["n_out"]

    def mk(block, n_pre, n_post):
        ex = edges_dict[block]["EX"]
        in_ = edges_dict[block]["IN"]
        gj = edges_dict[block]["GJ"]
        mod = FiuriSparseConn(n_pre, n_post,
                                    torch.stack([ex["src"], ex["dst"]], dim=0),
                                    torch.stack([in_["src"], in_["dst"]], dim=0),
                                    torch.stack([gj["src"], gj["dst"]], dim=0))
        # Overwrite initial random weights with our initialized vectors
        if ex["w"].numel():
            mod.ex_w.data.copy_(ex["w"].to(device))
            mod.ex_w.data.mul_(2.0)
        if in_["w"].numel():
            mod.in_w.data.copy_(in_["w"].to(device))
            mod.in_w.data.mul_(0.2)
        if gj["w"].numel():
            mod.gj_w.data.copy_(gj["w"].to(device))
        if device is not None:
            mod.to(device)
        return mod

    m_in2hid = mk("in2hid", n_in,  n_hid)
    m_hid    = mk("hid",    n_hid, n_hid)
    m_hid2out= mk("hid2out",n_hid, n_out)
    return m_in2hid, m_hid, m_hid2out

def build_twc(action_decoder: Callable,
              use_json_w: bool = False,) -> nn.Module:
    """ Extracts the data from thr TWC description
    and returns a nn.Module with the TWC implementation
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(json_path, "r") as f:
        net_data = json.load(f)

    edges = build_tw_edges(
        net_data,
        device=dev,
        dtype=torch.float32,
        init="kaiming_uniform",
        gain=1.0,
        random_seed=42,
        sort_by_dst=True,
        use_json_weights=use_json_w
    )
    sizes = edges["sizes"]
    n_in, n_hid, n_out = sizes["n_in"], sizes["n_hid"], sizes["n_out"]

    in2hid, hid, hid2out = make_sparse_modules_from_tw_edges(edges, device=dev)
    in_layer =  create_layer(n_in).to(dev)
    hid_layer = create_layer(n_hid).to(dev)
    out_layer = create_layer(n_out).to(dev)

    class TWC (nn.Module):
        """  
        When creating the module a proper input decoder to the 4 sensory neurons must be provided
        if you plan to pass raw observations. Decoder must accept keyword args
        n_inputs and device (compatible with utils.twc_io_wrapper.default_obs_encoder).
     
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.in_layer = in_layer
            self.hid_layer = hid_layer
            self.out_layer = out_layer
            self.in2hid = in2hid
            self.hid2hid = hid
            self.hid2out = hid2out
            self.decoder = action_decoder

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
            # Empty GJ bundle (no sensory gap junction graph)
            empty = torch.zeros(0, dtype=torch.long, device=device)
            gj_bundle = (empty, empty, torch.zeros(0, dtype=ex_in.dtype, device=device))
            o_pre_zero = torch.zeros(B, self.in_layer.num_cells, dtype=ex_in.dtype, device=device)
            _ = self.in_layer(ex_in, in_in, gj_bundle, o_pre_zero)

            # input -> hidden
            ex_raw, in_raw, gj_bundle = self.in2hid(self.in_layer.out_state)
            h = self.hid_layer(ex_raw, in_raw, gj_bundle, self.in_layer.out_state)

            # hidden -> hidden (one recurrent step)
            ex_h, in_h, gj_h = self.hid2hid(h)
            h = self.hid_layer(ex_h, in_h, gj_h, self.hid_layer.out_state)

            # hidden -> output
            ex_o, in_o, gj_o = self.hid2out(h)
            y = self.out_layer(ex_o, in_o, gj_o, self.hid_layer.out_state)  # presyn = hidden O

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

    return TWC()
