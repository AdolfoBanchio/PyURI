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
from PyURI_module import FIURIModule
import matplotlib.pyplot as plt


time_steps = 50

# Create FIURI node
fiuri = FIURIModule(
    num_cells=1,
    initial_in_state=0.0,
    initial_out_state=0.0,
    initial_threshold=1.0,
    initial_decay=0.1,
    learn_threshold=False,
    learn_decay=False,
    clamp_min=-10.0,
    clamp_max=10.0,
)


# Record states
in_states = torch.zeros(time_steps, dtype=torch.float32)
out_states = torch.zeros(time_steps, dtype=torch.float32)
influence_trace = torch.zeros(time_steps, dtype=torch.float32)

# Run simulation
print("Running simulation...")

for t in range(time_steps):
    print(f"==== Time step {t} ====")

    if t<10:
        inh_current=0.5 
        exc_current=0.0
        gj_current=0.0
    elif t<20:
        inh_current=0.0
        exc_current=1.5 
        gj_current=0.0
    elif t<30:
        inh_current=0.0
        exc_current=0.0
        gj_current=1.5
    else:
        inh_current=0.0
        exc_current=0.0
        gj_current=0.0

    # Prepare inputs for FIURIModule.forward
    # Shapes: (B=1, n=1)
    ex_raw = torch.tensor([[exc_current]], dtype=torch.float32)
    in_raw = torch.tensor([[inh_current]], dtype=torch.float32)
    # No GJ edges in this simple test
    gj_bundle = (
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.float32),
    )
    # Presyn outputs for GJ sign (not used here but required by API)
    o_pre = torch.full((1, 1), fill_value=gj_current,dtype=torch.float32)
    # Step FIURI module
    out = fiuri.forward(ex_raw, in_raw, gj_bundle, o_pre)
    influence_trace[t] = out.item()

    output_state = fiuri.out_state.item()
    internal_state = fiuri.in_state.item()
    # Record states
    in_states[t] = internal_state
    out_states[t] = output_state

# Plot results:
# plot 1: internal state and out state vs time steps
# plot 2: input currents vs time steps
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(in_states.numpy(), label='Internal State (E)', color='blue')
ax1.plot(out_states.numpy(), label='Output State (S)', color='orange')
ax1.axhline(y=fiuri.threshold.item(), color='red', linestyle='--', label='Threshold (T)')
ax1.set_title('PyURI Neuron Dynamics')
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('State Value')
ax1.legend()
ax1.grid()

ax2.plot(influence_trace.numpy())
ax2.set_title('Input Currents')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Current Value')
ax2.legend(['Stimulus (S)'])
ax2.grid()

plt.tight_layout()
plt.show()
save_path = os.path.join(os.path.dirname(__file__), "out/pyuri_dynamics.png")

plt.savefig(save_path)
