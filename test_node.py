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
import torch
from typing import Optional, Iterable, Union
from FIURI_node import FIURI_node
import matplotlib.pyplot as plt

time_steps = 50

# Create FIURI node
fiuri = FIURI_node(
    n=1,
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
in_states = torch.zeros(time_steps)
out_states = torch.zeros(time_steps)
current_trace = torch.zeros((time_steps,3))
# Run simulation
print("Running simulation...")

for t in range(time_steps):
    if t<10:
        inh_current = -0.5
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

    currents = [inh_current, exc_current, gj_current]
    current = torch.tensor(sum(currents)).view(1,1)  # shape (batch, n)
    
    # save each current value for plotting
    current_trace[t] = torch.tensor(currents)
    
    # Step FIURI node
    fiuri.forward(current)
    # Record states
    in_states[t] = fiuri.in_state.item()
    out_states[t] = fiuri.s.item()

# Plot results:
# plot 1: internal state and out state vs time steps
# plot 2: input currents vs time steps
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(in_states.numpy(), label='Internal State (E)', color='blue')
ax1.plot(out_states.numpy(), label='Output State (S)', color='orange')
ax1.axhline(y=fiuri.threshold.item(), color='red', linestyle='--', label='Threshold (T)')
ax1.set_title('FIURI Neuron Dynamics')
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('State Value')
ax1.legend()
ax1.grid()

ax2.plot(current_trace.numpy())
ax2.set_title('Input Currents')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Current Value')
ax2.legend(['Inhibitory', 'Excitatory', 'Gap Junction'])
ax2.grid()
plt.tight_layout()
plt.show()
plt.savefig('fiuri_dynamics.png')
