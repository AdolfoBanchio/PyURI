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
        initial_threshold=0,
        initial_decay=0.2,
        learn_threshold=True,
        learn_decay=True,
        clamp_min=-10.0,
        clamp_max=10.0,
        sum_input=True,
        debug=False
    )

def cad_connection(net: Network, src_n, dst_n, conn_W, conn_mask=None):
    """  
        Creates and ADDS connection to network.
    """
    src_layer = net.layers[src_n]
    dst_layer = net.layers[dst_n]

    conn = FIURI_Connection(source=src_layer,
                      target=dst_layer,
                      w=conn_W)

    # If a binary mask is provided, prune (zero) disallowed weights and
    # remove the reparam so conn.w is a plain Parameter with zeros baked in.
    # Mask semantics: 1/True -> keep, 0/False -> prune to zero.
    if conn_mask is not None:
        # Ensure mask is float/bool tensor on same device/dtype
        if conn_mask.dtype != torch.bool and conn_mask.dtype != torch.uint8:
            mask = conn_mask.to(dtype=conn.w.dtype)
        else:
            mask = conn_mask
        # torch.nn.utils.prune expects the mask to be registered on the module
        prune.custom_from_mask(conn, name='w', mask=mask)
        prune.remove(conn, 'w')
    
    net.add_connection(connection=conn,
                       source=src_n,
                       target=dst_n)


def build_TWC() -> Network:

    net = Network()

    with open(json_path, "r") as f:
        net_data = json.load(f)

    gropus = net_data["groups"]
    matrices = build_tw_matrices(net_data)

    for layer in gropus.keys():
        n = len(gropus[layer])
        l_name = layer
        net.add_layer(create_layer(n),
                      name=l_name)

    for conn in matrices["connections"]:
        source_n = conn["source"]  # layer names
        target_n = conn["target"]
        W = conn["weight"]         # weight matrix
        M = conn.get("mask", None) # boolean/float mask

        cad_connection(net, source_n, target_n, W, M)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    return net
    
