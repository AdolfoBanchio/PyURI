""" 
    This module creates one simple FIURI_node instance
    and tests its methods and grapihcs the dinamics of its variables.
    for a simple case with fixed threshold and decay factor.
    T = 1.0
    DF = 0.1

    The neuron is stimulated with 3 constant currents:
        - 0.5 for 10 time steps (excitatory)
        - 1.5 for 10 time steps (gap junction)
        - -1.0 for 10 time steps (inhibitory)
    and then left to relax for 20 time steps.

    weights fixed to 1.0 for simplicity.
    Plot the internal and output sates of the neuron.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from typing import Optional, Iterable, Union
from FIURI_node import FIURI_node
import matplotlib.pyplot as plt
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.topology_features import Weight, Bias
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages

# one neuron as source, 3 channels because has 3 junctions with target 
source_layer = Input(shape=(1,3)) 

num_cells = 1
# Create FIURI node
fiuri = FIURI_node(
    num_cells=num_cells,
    initial_in_state=0.0,
    initial_out_state=0.0,
    initial_threshold=1.0,
    initial_decay=0.1,
    learn_threshold=False,
    learn_decay=False,
    clamp_min=-10.0,
    clamp_max=10.0,
    sum_input=True,
    debug=True
)
# Identity connection: input channel k -> Fiuri channel k
# Shapes are flattened for weights
W = torch.eye(3) # 3x3 identity
con = Connection(
        source=source_layer,
        target=fiuri,
        w=W
)

# Build network
net = Network()

net.add_layer(
    layer=source_layer,
    name="in"
)

net.add_layer(
    layer=fiuri,
    name="fiuri"
)

net.add_connection(
    connection=con,
    source="in",
    target="fiuri"
)

in_state_monitor = Monitor(
    obj=fiuri,
    state_vars=["in_state"]
)

out_state_monitor = Monitor(
    obj=fiuri,
    state_vars=["out_state"]
)

net.add_monitor(out_state_monitor, name="mon_O")
net.add_monitor(in_state_monitor, name="mon_E")

time_steps = 50
batch = 1

# Generate trace of inputs to run the network 
X = torch.zeros(time_steps, batch, num_cells, 3, dtype=torch.float32)  # (T, B, 1, 3)
X[0:10, 0, 0, :] = torch.tensor([0.0, 0.5, 0.0])   # [exc, inh, gj]
X[10:20,0, 0, :] = torch.tensor([1.5, 0.0, 0.0])
X[20:30,0, 0, :] = torch.tensor([0.0, 0.0, 1.5])

print(X)
# Record states
in_states = torch.zeros(time_steps, dtype=torch.float32)
out_states = torch.zeros(time_steps, dtype=torch.float32)
influence_trace = torch.zeros(time_steps, dtype=torch.float32)

# Run simulation
print("Running simulation...")
net.run(inputs={"in": X}, time=time_steps, one_step=True)  # input_time_dim defaults to 0

in_state = in_state_monitor.get("in_state")
out_state = out_state_monitor.get("out_state")


fig = plt.figure(figsize=(15,6))
plt.plot(in_state.view(-1), label='Internal State (E)', color='blue')
plt.plot(out_state.view(-1), label='Output State (S)', color='orange')
plt.axhline(y=fiuri.threshold.item(), color='red', linestyle='--', label='Threshold (T)')
plt.title('PyURI Neuron Dynamics')
plt.xlabel('Time Steps')
plt.ylabel('State Value')
plt.legend()
plt.grid()

plt.show()

save_path = os.path.join(os.path.dirname(__file__), "out/pyuri_dynamics_integration.png")
plt.savefig(save_path)
