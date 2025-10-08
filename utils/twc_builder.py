import json
import os
import torch
from torch import nn
from utils.w_builder import build_tw_edges
from PyURI_module import FIURIModule, FiuriSparseConn

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
        initial_decay=2.2,
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
        if in_["w"].numel():
            mod.in_w.data.copy_(in_["w"].to(device))
        if gj["w"].numel():
            mod.gj_w.data.copy_(gj["w"].to(device))
        if device is not None:
            mod.to(device)
        return mod

    m_in2hid = mk("in2hid", n_in,  n_hid)
    m_hid    = mk("hid",    n_hid, n_hid)
    m_hid2out= mk("hid2out",n_hid, n_out)
    return m_in2hid, m_hid, m_hid2out

def build_twc() -> nn.Module:
    """ 
    Extracts the data from thr TWC description
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
    )
    sizes = edges["sizes"]
    n_in, n_hid, n_out = sizes["n_in"], sizes["n_hid"], sizes["n_out"]

    in2hid, hid, hid2out = make_sparse_modules_from_tw_edges(edges, device=dev)
    in_layer =  create_layer(n_in).to(dev)
    hid_layer = create_layer(n_hid).to(dev)
    out_layer = create_layer(n_out).to(dev)

    class TWC (nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.in_layer = in_layer
            self.hid_layer = hid_layer
            self.out_layer = out_layer
            self.in2hid = in2hid
            self.hid2hid = hid
            self.hid2out = hid2out

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            One TWC step.
            x: (B, 2, n_in) -> channel 0 = desired internal E, channel 1 = desired output O for input neurons.
            """
            # seed input layer state directly, then one neuron step without external input
            self.in_layer.set_internal_state(x[:, 0, :])
            self.in_layer.set_output_state(x[:, 1, :])
            self.in_layer.neuron_step()  # uses current E

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
        
        def reset_state_variables(self):
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
