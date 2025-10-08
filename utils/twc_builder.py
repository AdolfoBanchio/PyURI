import sys
import os
import json
import torch
from FIURI_node import FIURI_node, FIURI_Connection
from utils.w_builder import build_tw_matrices
from bindsnet.network import Network
from bindsnet.network.topology import Connection
import torch.nn.utils.prune as prune

""" 
    Reads the TWC structure from .json file
    and constructs the Bindsnet netwrok using FIURI nodes. 
"""
json_path = os.path.join(os.path.dirname(__file__),"TWC_fiu.json")

def create_layer(N_neurons) -> FIURI_node:
    return FIURI_node(
        num_cells=N_neurons,
        initial_in_state=0.0,
        initial_out_state=0.0,
        initial_threshold=0.3,
        initial_decay=2.2,
        learn_threshold=True,
        learn_decay=True,
        clamp_min=-10.0,
        clamp_max=10.0,
        sum_input=True,
        debug=False
    )

def cad_connection(net: Network, src_n, dst_n, conn_W, conn_mask=None, device=None):
    src_layer = net.layers[src_n]
    dst_layer = net.layers[dst_n]

    # 0) pick a target device
    if device is None:
        try:
            device = next(src_layer.parameters()).device
        except StopIteration:
            device = next(net.parameters()).device

    # 1) ensure weight tensor is on the right device
    if not isinstance(conn_W, torch.Tensor):
        conn_W = torch.tensor(conn_W, dtype=torch.float32, device=device)
    else:
        conn_W = conn_W.to(device=device, dtype=torch.float32)

    # 2) build connection (w is a registered Parameter on 'device')
    conn = FIURI_Connection(source=src_layer, target=dst_layer, w=conn_W)

    # 3) apply pruning mask (keep reparam; DO NOT prune.remove)
    if conn_mask is not None:
        mask = torch.as_tensor(conn_mask, dtype=torch.bool, device=device)
        if mask.shape != conn.w.shape:
            raise ValueError(f"Mask shape {mask.shape} != weight shape {tuple(conn.w.shape)}")
        torch.nn.utils.prune.custom_from_mask(conn, name="w", mask=mask)
        # conn now has w_orig (Parameter) and a mask buffer registered

    # 4) EXPLICITLY move the connection module (params + buffers + parametrizations)
    conn.to(device)

    # 5) sanity assert
    eff = getattr(conn, "w", None)
    if eff is None:
        raise RuntimeError("Connection has no attribute 'w' after pruning.")
    if eff.device != device:
        raise RuntimeError(f"Effective weight on {eff.device}, expected {device}")

    # 6) register in the net
    net.add_connection(connection=conn, source=src_n, target=dst_n)



def build_TWC() -> Network:
    net = Network()

    with open(json_path, "r") as f:
        net_data = json.load(f)

    groups = net_data["groups"]
    matrices = build_tw_matrices(net_data)

    for layer_name, neuron_list in groups.items():
        net.add_layer(create_layer(len(neuron_list)), name=layer_name)

    for conn in matrices["connections"]:
        source_n = conn["source"]  # layer names
        target_n = conn["target"]
        W = conn["weight"]         # weight matrix
        M = conn.get("mask", None) # boolean/float mask

        cad_connection(net, source_n, target_n, W, M)
    
    return net
    
