import json
import os
import math
from typing import Callable
import torch
from torch import nn
from fiuri import FIURIModule, FiuriDenseConn, FiuriSparseGJConn, FIURIModuleV2
from .w_builder import build_tw_matrices
from twc.twc import TWC

json_path = os.path.join(os.path.dirname(__file__), "TWC_fiu.json")

def build_twc(obs_encoder: Callable,
              action_decoder: Callable,
              internal_steps: int,
              initial_thresholds: list[float] = [ -0.5, 0.0, 0.0 ],
              initial_decays: list[float] = [2.2, 0.1, 0.1],
              rnd_init: bool = False,
              use_V2: bool = False,
              log_stats: bool = True,
              **kwargs) -> TWC:
    """ Extracts the data from thr TWC description
    and returns a nn.Module with the TWC implementation
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(json_path, "r") as f:
        net_data = json.load(f)

    masks, sizes = build_tw_matrices(net_data)

    n_in, n_hid, n_out = sizes["n_in"], sizes["n_hid"], sizes["n_out"]

    Layer = FIURIModuleV2 if use_V2 else FIURIModule
    
    in_layer =  Layer(
        num_cells=n_in,
        initial_in_state=0.0,
        initial_out_state=0.0,
        initial_threshold=initial_thresholds[0],
        initial_decay=initial_decays[0],
        clamp_min=-10.0,
        clamp_max=10.0,
        rnd_init=rnd_init,
        **kwargs
    )
    hid_layer = Layer(
        num_cells=n_hid,
        initial_in_state=0.0,
        initial_out_state=0.0,
        initial_threshold=initial_thresholds[1],
        initial_decay=initial_decays[1],
        clamp_min=-10.0,
        clamp_max=10.0,
        rnd_init=rnd_init,
        **kwargs
    )
    out_layer = Layer(
        num_cells=n_out,
        initial_in_state=0.0,
        initial_out_state=0.0,
        initial_threshold=initial_thresholds[2],
        initial_decay=initial_decays[2],
        clamp_min=-10.0,
        clamp_max=10.0,
        rnd_init=rnd_init,
        **kwargs
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

    return TWC(in_layer=in_layer, 
               hid_layer=hid_layer, 
               out_layer=out_layer, 
               in2hid_IN=in2hid, 
               in2hid_GJ=gj_conn, 
               hid_IN=hid_IN, 
               hid_EX=hid_EX, 
               hid2out_EX=hid2out_EX, 
               obs_encoder=obs_encoder, 
               action_decoder=action_decoder, 
               internal_steps=internal_steps, 
               log_stats=log_stats)
