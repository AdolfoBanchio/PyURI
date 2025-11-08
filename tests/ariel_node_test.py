import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import ariel.Neuron as neu
import ariel.Connection as con
import matplotlib.pyplot as plt
""" 
    This module creates one simple neuron instance.
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
"""
inh_neuron = neu.Neuron("in")
inh_neuron.outputstate=0

exc_neuron = neu.Neuron("exc")
exc_neuron.outputstate=0

gj_neuron = neu.Neuron("gj")
gj_neuron.outputstate=0

target_neuron = neu.Neuron("N1")
target_neuron.initialize(1.0,0.0,0.0,0.1)


inh_con = con.Connection(con.ConnectionType.ChemIn, inh_neuron, target_neuron, 1.0) # inhibitory
ex_con = con.Connection(con.ConnectionType.ChemEx, exc_neuron, target_neuron, 1.0) # excitatory
gj_con = con.Connection(con.ConnectionType.AGJ, gj_neuron, target_neuron, 1.0) # gap junction

denditric_inputs = [inh_con, ex_con, gj_con]

time_steps = 50
output_trace = []
internal_trace = []
current_trace = []

for t in range(time_steps):
    print(f"==== Time step {t} ====")
    if t<10:
        inh_neuron.outputstate=0.5 
        exc_neuron.outputstate=0.0 
        gj_neuron.outputstate=0.0 
    elif t<20:
        inh_neuron.outputstate=0.0
        exc_neuron.outputstate=1.5 
        gj_neuron.outputstate=0.0
    elif t<30:
        inh_neuron.outputstate=0.0
        exc_neuron.outputstate=0.0
        gj_neuron.outputstate=1.5
    else:
        inh_neuron.outputstate=0.0
        exc_neuron.outputstate=0.0
        gj_neuron.outputstate=0.0

    curr_stimulus = target_neuron.computeVnext(denditric_inputs)
    current_trace.append(curr_stimulus)
    target_neuron.commitComputation()

    output_state = target_neuron.outputstate
    internal_state = target_neuron.internalstate
    print('Output state (O):', output_state)
    print('Internal state (O):', internal_state)
    # store values for plotting
    output_trace.append(output_state)
    internal_trace.append(internal_state)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(internal_trace, label='Internal State (E)', color='blue')
ax1.plot(output_trace, label='Output State (O)', color='orange')
ax1.axhline(y=1, color='red', linestyle='--', label='Threshold (T)')
ax1.set_title('FIURI Neuron Dynamics')
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('State Value')
ax1.legend()
ax1.grid()

ax2.plot(current_trace)
ax2.set_title('Stimulus')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Current Value')
ax2.legend('Stimulus S')
ax2.grid()

plt.tight_layout()
plt.show()
plt.savefig('test_outs/fiuri_dynamics.png')

    
