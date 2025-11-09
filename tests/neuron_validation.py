import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import csv
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
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

    def __init__(self):
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

        self.E_trace = []
        self.O_Trace = []

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

        self.save_trace(self.neuron.internalstate, self.neuron.outputstate)
        return (self.neuron.internalstate, self.neuron.outputstate)
    
    def save_trace(self, E, O):
        self.E_trace.append(E)
        self.O_Trace.append(O)

class AuxTorchNeu:
    
    def __init__(self):
        self.neuron = fiuri = FIURIModule(num_cells=1,
                                          initial_in_state=0.0,
                                          initial_out_state=0.0,
                                          initial_threshold=1.0,
                                          initial_decay=0.1,
                                          clamp_min=-10.0,
                                          clamp_max=10.0
                                          )
        
        self.state = (0.0, 0.0)
        self.E_trace = []
        self.O_Trace = []
    
    @torch.no_grad()
    def step(self, inh_input: int, ex_input: int, gj_input: int) -> int | int:
        chem_influence = torch.tensor([[ex_input - inh_input]], dtype=torch.float32)
    
        # Calculate current stimulus (E + Ij)
        current_stimulus = self.state[0] + chem_influence.item()

        # Gap junction bundle: (src_idx, dst_idx, weights)
        gj_bundle = (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.float32),
        )

        # Presyn outputs for GJ sign
        o_pre = torch.full((1, 1), fill_value=gj_input, dtype=torch.float32)

        new_o, (E, O) = self.neuron.forward(chem_influence=chem_influence,
                                                state=self.state,
                                                gj_bundle=gj_bundle,
                                                o_pre=o_pre)
        self.save_trace(E.squeeze(), O.squeeze())
        
        self.state = (E, O)
        return self.state
    
    def save_trace(self, E, O):
        self.E_trace.append(E)
        self.O_Trace.append(O)
    

def plot_neuron_state(E_trace: list[int], O_trace: list[int], title: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(E_trace, label='Internal State (E)', color='blue')
    ax.plot(O_trace, label='Output State (O)', color='orange')
    ax.axhline(y=1, color='red', linestyle='--', label='Threshold (T)')
    ax.set_title(title)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('State Value')
    ax.legend()
    ax.grid()
    plt.show()
    return fig

def main():
    timesteps = 50
    a_neu = AuxArielNeu()
    t_neu = AuxTorchNeu()

    for t in range(timesteps):
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
            gj_current=np.random.uniform(-1, 1)
        else:
            inh_current= np.random.uniform(-1, 1)
            exc_current= np.random.uniform(-1, 1)
            gj_current= np.random.uniform(-1, 1)

        a_state = a_neu.step(inh_input=inh_current,
                   ex_input=exc_current,
                   gj_input=gj_current)
        
        print("Fiuri state: ", a_state)
        t_state = t_neu.step(inh_input=inh_current,
                   ex_input=exc_current,
                   gj_input=gj_current)
        print("PyUri state: ", t_state)
        
    # Convert lists to numpy arrays and subtract
    e_diff = np.array(a_neu.E_trace) - np.array(t_neu.E_trace)
    o_diff = np.array(a_neu.O_Trace) - np.array(t_neu.O_Trace)

    a_plot = plot_neuron_state(E_trace=a_neu.E_trace,
                               O_trace=a_neu.O_Trace,
                               title='Fiuri States Trace')
    a_plot_save_name = 'fiuri_neuron_staes.png'

    t_plot = plot_neuron_state(E_trace=np.array(t_neu.E_trace),
                               O_trace=np.array(t_neu.O_Trace),
                               title='PyUri States Trace')
    t_plot_save_name = 'pyuri_neuron_states.png'
    
    # Create the output directory if it doesn't exist
    save_path = os.path.join(Path(__file__).parents[1], 'out', 'tests', 'neuron_validation')
    os.makedirs(save_path, exist_ok=True)

    # Save the plots
    a_plot.savefig(os.path.join(save_path, a_plot_save_name))
    t_plot.savefig(os.path.join(save_path, t_plot_save_name))

if __name__ == "__main__":
    main()