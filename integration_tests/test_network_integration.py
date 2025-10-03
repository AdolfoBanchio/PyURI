import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from typing import Optional, Iterable, Union
from FIURI_node import FIURI_node
from w_builder import build_tw_matrices
from bindsnet.network import Network
from bindsnet.network.nodes import Input
from bindsnet.network.topology import Connection
from bindsnet.network.topology_features import Weight, Bias
from bindsnet.analysis.visualization import summary

""" 
    Try to define the Tap Withdrawal Circuit as a bindsnet network.
    With the following structure:
        - layer 1: 4 sensory neurons.
        - layer 2: 5 inner neurons.
        - layer 3: 2 output motor neurons.

    Then run some inptut to the network and try to graph the results.
"""
net_file_name = "TWC_fiu.json"
net_file_path = os.path.join(os.path.dirname(__file__),net_file_name)

with open(net_file_path, "r") as f:
    net_data = json.load(f)

matrices = build_tw_matrices(net_data)

N_in = 4
N_hid = 5
N_out = 2

in_layer = FIURI_node(
    num_cells=N_in,
    initial_in_state=0.0,
    initial_out_state=0.0,
    initial_threshold=1.0,
    initial_decay=0.1,
    learn_threshold=True,
    learn_decay=True,
    clamp_min=-10.0,
    clamp_max=10.0,
    sum_input=True,
    debug=False
)

hid_layer = FIURI_node(
    num_cells=N_hid,
    initial_in_state=0.0,
    initial_out_state=0.0,
    initial_threshold=1.0,
    initial_decay=0.1,
    learn_threshold=True,
    learn_decay=True,
    clamp_min=-10.0,
    clamp_max=10.0,
    sum_input=True,
    debug=False
)

out_layer = FIURI_node(
    num_cells=N_out,
    initial_in_state=0.0,
    initial_out_state=0.0,
    initial_threshold=1.0,
    initial_decay=0.1,
    learn_threshold=True,
    learn_decay=True,
    clamp_min=-10.0,
    clamp_max=10.0,
    sum_input=True,
    debug=False
)

net = Network()

net.add_layer(in_layer, name="in")
net.add_layer(hid_layer, name="hid")
net.add_layer(out_layer, name="out")

net.add_connection(
    Connection(source=in_layer,
               target=hid_layer, 
               w=matrices["W_in2hid_EX"]), 
    source="in", target="hid"
    )
net.add_connection(
    Connection(source=in_layer,
               target=hid_layer, 
               w=matrices["W_in2hid_IN"]), 
    source="in", target="hid"
    )
net.add_connection(
    Connection(source=in_layer,
               target=hid_layer, 
               w=matrices["W_in2hid_GJ"]), 
    source="in", target="hid"
    )

net.add_connection(
    Connection(source=hid_layer,
               target=hid_layer, 
               w=matrices["W_hid_EX"]), 
    source="hid", target="hid"
    )
net.add_connection(
    Connection(source=hid_layer,
               target=hid_layer, 
               w=matrices["W_hid_IN"]), 
    source="hid", target="hid"
    )
net.add_connection(
    Connection(source=hid_layer,
               target=hid_layer, 
               w=matrices["W_hid_GJ"]), 
    source="hid", target="hid"
    )

net.add_connection(
    Connection(source=hid_layer,
               target=out_layer, 
               w=matrices["W_hid2out_EX"]), 
    source="hid", target="out"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

print(summary(net=net))

print("\nNetwork Parameters:\n")
for name, param in net.named_parameters():
    print(f"Name: {name}")
    print(f"  Shape: {param.shape}")
    print(f"  Dtype: {param.dtype}")
    print(f"  Device: {param.device}")
    print(f"  Requires grad: {param.requires_grad}")
    print(f"  Data (first 5 elements): {param.flatten()[:5].tolist()}")
    print("-" * 40)
 
