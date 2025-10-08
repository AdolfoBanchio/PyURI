import torch 
import torch.nn as nn
import numpy as np
from typing import Optional

class FIURI_module(nn.Module):
    """  
    Nerual layer of the FIURI model. (fully pytorch)
    Sn = En + sum(I_jin) j=1..m where m is ammount of connections(9) 
    
    On = ( Sn- Tn if Sn > Tn 
         ( 0 other case (10) 
    
        | Sn - Tn if Sn > Tn
    En =  En - dn if Sn ≤ Tn and Sn = En  (11) 
        \ Sn other case  
         
    Ij in = 
            ωj * Oj if Oj ≥ En y gap junct. 
            -ωj * Oj if Oj < En y gap junct. 
            ωj * Oj chemical excitatory 
            -ωj * Oj chemical inhibitory
    where Oj is the output state of the presynaptic neuron j and ωj is the weight of the connection from neuron j to neuron n.
    
    where:
        - En and On represent the internal state and the output state of neuron n, respectively. (not learnable)
        - Sn represents the stimiulus comming through the connections due to the currents Ij_in.
        - Eqs 10,11 represent the dynamic of the neuron.
        - Tn and dn are the neuronal parameters that have to be learned and represent:
            - Tn the firing threshold
            - dn the decay factor (due to not enough stimulus)
    
    Each neuron has 3 channels for communication with FIURI_connections
        - 0: EX
        - 1: IN
        - 2: GJ
    """
    def __init__(
        self,
        debug: Optional[bool] = False,
        num_cells: Optional[int] = None,
        initial_in_state: Optional[float] = 0.0,   # scalar default
        initial_out_state: Optional[float] = 0.0,  # scalar default
        initial_threshold: Optional[float] = 1.0,
        initial_decay: Optional[float] = 0.1,
        learn_threshold: bool = True,
        learn_decay: bool = True,
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
        **kwargs,
    ) -> None:
        
        super().__init__()

        assert (num_cells is not None), "Must provide number of neurons per layer"

        