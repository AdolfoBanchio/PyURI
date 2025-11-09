import sys
import json
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import os
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from twc.twc_builder import build_twc
from twc.twc_io import mcc_obs_encoder, twc_out_2_mcc_action, POS_MAX, POS_MIN, VEL_MAX
from ariel.Model import Model as FiuModel

ENV = "MountainCarContinuous-v0"
SEED = 42

def main():
    env = gym.make(ENV)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    out_dir = os.path.join('out/tests/twc_validation')
    os.makedirs(out_dir, exist_ok=True)

    # Build TWC for MCC with logging enabled
    twc = build_twc(
        obs_encoder=mcc_obs_encoder,
        action_decoder=twc_out_2_mcc_action,
        internal_steps=1,
        log_stats=True,
    )
    twc._set_all_weights_to_one()
    twc.eval()

    xml_path = os.path.join(Path(__file__).parent, 'TWFiuriBaseFIU.xml')
    fiu_twc = FiuModel('FIU')
    fiu_twc.loadFromFile(xml_path)
    fiu_twc.Reset()

    print(twc.state_dict())
    print(fiu_twc)

    with open(SRC_ROOT / "twc" / "TWC_fiu.json", "r", encoding="utf-8") as spec_file:
        groups_spec = json.load(spec_file)["groups"]
    group_order = ("input", "hidden", "output")
    neuron_groups = {layer: groups_spec[layer] for layer in group_order}
    fiu_layer_traces = {
        layer: {"in": [], "out": []}
        for layer in group_order
    }

    # Generate a short synthetic observation sequence (position, velocity)
    # Cover both sides of valley and velocities in [-VEL_MAX, VEL_MAX]
    T = 10
    a_actions = []
    t_actions = []
    with torch.no_grad():
        for t in range(T):
            obs = env.observation_space.sample()
            obs_t = torch.as_tensor(obs).unsqueeze(0)
            a = twc(obs_t)  # logs internal states each call
            t_actions.append(a.squeeze().item())
            a = fiu_twc.Update([obs[0], obs[1]])
            a_actions.append(a)
            for layer_name in group_order:
                layer_neurons = neuron_groups[layer_name]
                in_vals = []
                out_vals = []
                for neuron_name in layer_neurons:
                    neuron = fiu_twc.neuralnetwork.getNeuron(neuron_name)
                    in_vals.append(neuron.getInternalState())
                    out_vals.append(neuron.getOutputState())
                fiu_layer_traces[layer_name]["in"].append(torch.tensor(in_vals, dtype=torch.float32))
                fiu_layer_traces[layer_name]["out"].append(torch.tensor(out_vals, dtype=torch.float32))

    # Extract monitor logs into (T, N) tensors per layer
    monitor = twc.monitor
    layers = ['in', 'hid', 'out']
    series = {}
    for L in layers:
        in_states = torch.stack([step['in_state'][0] for step in monitor[L]], dim=0)  # (T, N)
        out_states = torch.stack([step['out_state'][0] for step in monitor[L]], dim=0)  # (T, N)
        series[L] = (in_states, out_states)

    # Collect equivalent traces for the author's implementation grouped per anatomical layer
    fiu_series = {}
    for layer_name in group_order:
        layer_in = torch.stack(fiu_layer_traces[layer_name]["in"], dim=0)
        layer_out = torch.stack(fiu_layer_traces[layer_name]["out"], dim=0)
        fiu_series[layer_name] = (layer_in, layer_out)

    print(a_actions)
    print(t_actions)


if __name__ == "__main__":
    main()
