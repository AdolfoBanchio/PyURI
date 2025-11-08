import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import csv
import os
import matplotlib.pyplot as plt
import torch
import ariel.Neuron as neu
import ariel.Connection as con
import matplotlib.pyplot as plt
from fiuri import FIURIModule

"""  
This test intention is to verify that the pytorch neuron implementation of the Fiuri 
dynamic is correct. Comparing it against the original Authors implementation.

"""

# Helper Class
class AuxArielNeu:

    def __init__(self, name):
        self.inh_neuron = neu.Neuron("in")
        self.inh_neuron.outputstate=0

        self.exc_neuron = neu.Neuron("exc")
        self.exc_neuron.outputstate=0

        self.gj_neuron = neu.Neuron("gj")
        self.gj_neuron.outputstate=0

        self.neuron = neu.Neuron("N1")
        self.neuron.initialize(1.0,0.0,0.0,0.1)

        inh_con = con.Connection(con.ConnectionType.ChemIn, self.inh_neuron, self.neuron, 1.0) # inhibitory
        ex_con = con.Connection(con.ConnectionType.ChemEx, self.exc_neuron, self.neuron, 1.0) # excitatory
        gj_con = con.Connection(con.ConnectionType.AGJ, self.gj_neuron, self.neuron, 1.0) # gap junction

        self.denditric_inputs = [inh_con, ex_con, gj_con]

    def step(self, inh_input: int, ex_input: int, gj_input: int) -> int | int:
        """  
            Recives the corresponing inputs to the neuron, and returns
            the internal and output state after the neuron dynamic step
        """
        self.inh_neuron.outputstate = inh_input
        self.exc_neuron.outputstate = ex_input
        self.gj_neuron.outputstate  = gj_input
        """  
        Because this neurons are only used as interfaces. And a internal step
        is never performed. We can ensure that the output mantain as is 
        when the step of our target neuron is computed.
        """

        self.neuron.computeVnext(self.denditric_inputs)
        self.neuron.commitComputation()

        return self.neuron.internalstate, self.neuron.outputstate


def plot_neuron_state(E_trace: list[int], O_trace: list[int]):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(E_trace, label='Internal State (E)', color='blue')
    ax.plot(O_trace, label='Output State (O)', color='orange')
    ax.axhline(y=1, color='red', linestyle='--', label='Threshold (T)')
    ax.set_title('FIURI Neuron Dynamics')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('State Value')
    ax.legend()
    ax.grid()
    plt.show()
    return fig

fig = plot_neuron_state([0,1,2,3,4,5,6,7],[7,6,45,4,3,2,1])
plt.show()