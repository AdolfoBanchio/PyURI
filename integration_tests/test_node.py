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
from PyURI_module import FIURIModule
import matplotlib.pyplot as plt
import csv


time_steps = 50

# Create FIURI node
fiuri = FIURIModule(
    num_cells=1,
    initial_in_state=0.0,
    initial_out_state=0.0,
    initial_threshold=1.0,
    initial_decay=0.1,
    clamp_min=-10.0,
    clamp_max=10.0,
)


# Record states
in_states = torch.zeros(time_steps, dtype=torch.float32)
out_states = torch.zeros(time_steps, dtype=torch.float32)
influence_trace = torch.zeros(time_steps, dtype=torch.float32)

# CSV logging data
csv_data = []

# Initialize state
current_state = None

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

    # Get current internal state (E) before forward pass
    E_before = fiuri.in_state.item() if current_state is None else current_state[0].item()

    # Prepare inputs for FIURIModule.forward
    # API: forward(chem_influence, state=None, gj_bundle=None, o_pre=None)
    # chem_influence = excitatory - inhibitory (combined influence)
    # Shapes: (B=1, n=1)
    chem_influence = torch.tensor([[exc_current - inh_current]], dtype=torch.float32)
    
    # Calculate current stimulus (E + Ij)
    current_stimulus = E_before + chem_influence.item()
    
    # Gap junction bundle: (src_idx, dst_idx, weights)
    gj_bundle = (
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.float32),
    )
    
    # Presyn outputs for GJ sign
    o_pre = torch.full((1, 1), fill_value=gj_current, dtype=torch.float32)
    
    influence_trace[t] = current_stimulus + gj_current
    
    # Step FIURI module - returns (new_o, (new_e, new_o_state))
    new_o, new_state = fiuri.forward(chem_influence, state=current_state, gj_bundle=gj_bundle, o_pre=o_pre)
    
    # Update module's internal state buffers (needed for next iteration)
    fiuri.in_state = new_state[0]  # new internal state
    fiuri.out_state = new_state[1]  # new output state
    
    # Update state for next iteration
    current_state = new_state
    
    # Record states
    in_states[t] = new_state[0].item()  # new internal state
    out_states[t] = new_o.item()
    
    # Record CSV data
    csv_data.append({
        'time_step': t,
        'inh_inf': -inh_current,  # negative inhibitory influence
        'exc_inf': exc_current,   # excitatory influence
        'gj_inf': gj_current,      # gap junction influence
        'current_influence': chem_influence.item(),  # sum Ij (chem_influence)
        'current_stimulus': current_stimulus + gj_current,  # E + Ij (internal + all influences)
        'output_state_O': new_o.item(),  # Output state
        'internal_state_E': new_state[0].item()  # Internal state
    })

# Plot results:
# plot 1: internal state and out state vs time steps
# plot 2: input currents vs time steps
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(in_states.numpy(), label='Internal State (E)', color='blue')
ax1.plot(out_states.numpy(), label='Output State (S)', color='orange')
# Get threshold value (parameter is raw, needs to be converted with relu threshold behavior)
threshold_value = fiuri.threshold.item()
ax1.axhline(y=threshold_value, color='red', linestyle='--', label='Threshold (T)')
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

# Write CSV file
csv_path = os.path.join(os.path.dirname(__file__), "out/pyuri_dynamics.csv")
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['time_step', 'inh_inf', 'exc_inf', 'gj_inf', 'current_influence', 'current_stimulus', 'output_state_O', 'internal_state_E']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in csv_data:
        writer.writerow(row)

print(f"\nSimulation complete!")
print(f"Plot saved to: {save_path}")
print(f"CSV data saved to: {csv_path}")
