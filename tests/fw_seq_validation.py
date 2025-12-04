import json
import os
import math
import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import torch
import numpy as np

from twc.twc_builder import build_twc
from twc.twc_io import mcc_obs_encoder, twc_out_2_mcc_action

# Ariel reference implementation (non-PyTorch)
from ariel.NeuralNetwork import NeuralNetwork
from ariel.Neuron import Neuron
from ariel.Connection import Connection, ConnectionType


HERE = os.path.dirname(__file__)
TWC_JSON = os.path.join(os.path.dirname(HERE), 'src', 'twc', 'TWC_fiu.json')


def build_ariel_from_json(json_path: str,
                          init_th=( -0.5, 0.0, 0.0),
                          init_df=(  2.2, 0.1, 0.1)):
    with open(json_path, 'r') as f:
        spec = json.load(f)

    nn = NeuralNetwork('TWC_Ariel')

    groups = spec['groups']
    input_names  = groups['input']
    hidden_names = groups['hidden']
    output_names = groups['output']

    # Create neurons with 0 init states
    def add_neuron(name):
        n = Neuron(name)
        n.setInternalState(0.0)
        n.setOutputState(0.0)
        nn.neurons[name] = n
        return n

    for n in input_names + hidden_names + output_names:
        add_neuron(n)

    # Set thresholds/decays per layer (use test* as computeVnext uses them)
    for n in input_names:
        nn.neurons[n].setTestThreshold(init_th[0])
        nn.neurons[n].setTestDecayFactor(init_df[0])
    for n in hidden_names:
        nn.neurons[n].setTestThreshold(init_th[1])
        nn.neurons[n].setTestDecayFactor(init_df[1])
    for n in output_names:
        nn.neurons[n].setTestThreshold(init_th[2])
        nn.neurons[n].setTestDecayFactor(init_df[2])

    # Add connections per spec; default all weights to 1.0 for cross-impl parity
    for e in spec['edges']:
        src = nn.neurons[e['src']]
        dst = nn.neurons[e['dst']]
        ety = e['type']
        if ety == 'EX':
            ct = ConnectionType.ChemEx
        elif ety == 'IN':
            ct = ConnectionType.ChemIn
        elif ety == 'GJ':
            ct = ConnectionType.SGJ
        else:
            continue
        c = Connection(ct, src, dst, 1.0)
        c.setTestWeight(1.0)
        nn.connections.append(c)

    return nn, input_names, hidden_names, output_names


def ariel_step(nn: NeuralNetwork, input_names, obs_tensor: torch.Tensor):
    # Encode obs with same logic used by Torch side
    ex_in, in_in = mcc_obs_encoder(obs_tensor, n_inputs=4, device=obs_tensor.device)
    input_values = (ex_in + in_in).squeeze(0).detach().cpu().numpy()  # (4,)

    # Assign to input neurons: [PVD, PLM, AVM, ALM]
    for i, name in enumerate(input_names):
        n = nn.neurons[name]
        val = float(input_values[i])
        n.setInternalState(val)
        n.setOutputState(val)

    nn.doSimulationStep()  # updates hidden + output using old states


def collect_layer_states_ariel(nn: NeuralNetwork, names):
    E = np.array([nn.neurons[n].getInternalState() for n in names], dtype=np.float32)[None, :]
    O = np.array([nn.neurons[n].getOutputState()   for n in names], dtype=np.float32)[None, :]
    return E, O


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Build torch TWC (no V2)
    twc = build_twc(mcc_obs_encoder, twc_out_2_mcc_action, use_V2=False, rnd_init=False, log_stats=False)
    twc.to(device)
    twc.eval()

    # Force all learnable connection weights to 1 (softplus -> 1)
    if hasattr(twc, '_set_all_weights_to_one'):
        twc._set_all_weights_to_one()

    # Build Ariel reference and also set weights to 1
    nn, input_names, hidden_names, output_names = build_ariel_from_json(TWC_JSON)

    # Prepare a deterministic sequence
    T = 6
    obs_seq = torch.tensor([
        [-0.3,  0.00],
        [-0.7,  0.05],
        [ 0.1, -0.04],
        [ 0.5,  0.07],
        [-1.0, -0.09],
        [ 0.6,  0.00],
    ], dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,2)

    # Torch stepwise (BPTT) states
    state = twc.get_initial_state(batch_size=1, device=device, dtype=torch.float32)
    hid_E_seq, hid_O_seq, out_E_seq, out_O_seq = [], [], [], []
    a_stepwise = []
    for t in range(obs_seq.size(1)):
        a_t, state = twc.forward_bptt(obs_seq[:, t, :], state)
        a_stepwise.append(a_t)
        hid_E_seq.append(state['hid'][0].detach().cpu().numpy())
        hid_O_seq.append(state['hid'][1].detach().cpu().numpy())
        out_E_seq.append(state['out'][0].detach().cpu().numpy())
        out_O_seq.append(state['out'][1].detach().cpu().numpy())

    a_stepwise = torch.stack(a_stepwise, dim=1)

    # Torch forward_sequence actions should match stepwise
    state0 = twc.get_initial_state(batch_size=1, device=device, dtype=torch.float32)
    a_seq, _ = twc.forward_sequence(obs_seq, state0)
    print(a_seq)
    assert torch.allclose(a_seq, a_stepwise, atol=1e-6), 'forward_sequence y paso a paso no coinciden'

    # Ariel stepwise and compare states per t
    atol = 1e-4
    rtol = 1e-5
    for t in range(obs_seq.size(1)):
        ariel_step(nn, input_names, obs_seq[:, t, :])
        a_hid_E, a_hid_O = collect_layer_states_ariel(nn, hidden_names)
        a_out_E, a_out_O = collect_layer_states_ariel(nn, output_names)

        # Compare to torch states at same time step
        th_hid_E = hid_E_seq[t]
        th_hid_O = hid_O_seq[t]
        th_out_E = out_E_seq[t]
        th_out_O = out_O_seq[t]

        np.testing.assert_allclose(a_hid_E, th_hid_E, rtol=rtol, atol=atol,
                                   err_msg=f"Hidden E mismatch at t={t}")
        np.testing.assert_allclose(a_hid_O, th_hid_O, rtol=rtol, atol=atol,
                                   err_msg=f"Hidden O mismatch at t={t}")
        np.testing.assert_allclose(a_out_E, th_out_E, rtol=rtol, atol=atol,
                                   err_msg=f"Out E mismatch at t={t}")
        np.testing.assert_allclose(a_out_O, th_out_O, rtol=rtol, atol=atol,
                                   err_msg=f"Out O mismatch at t={t}")

    print('fw_seq_validation: OK')


if __name__ == '__main__':
    # Allow running directly
    main()

