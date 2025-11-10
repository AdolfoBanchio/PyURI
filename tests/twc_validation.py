import sys
import json
import math
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import gymnasium as gym
from twc.twc_builder import build_twc
from twc.twc_io import (
    mcc_obs_encoder,
    twc_out_2_mcc_action,
    POS_MAX,
    POS_MIN,
    VEL_MAX,
    MIN_STATE,
    MAX_STATE,
    POS_VALLEY_VAL,
    VEL_VALLEY_VAL,
)
from ariel.Model import Model as FiuModel
from ariel import Connection as con

ENV = "MountainCarContinuous-v0"
SEED = 42

def extract_ariel_neuron_states(fiu_model):
    """Extract all neuron internal and output states from ariel model."""
    states = {}
    for name in fiu_model.neuralnetwork.getNeuronNames():
        neuron = fiu_model.getNeuron(name)
        states[name] = {
            'internal': neuron.getInternalState(),
            'output': neuron.getOutputState(),
            'threshold': neuron.getTestThreshold(),
            'decay': neuron.getTestDecayFactor(),
        }
    return states

def extract_ariel_connection_weights(fiu_model):
    """Extract all connection weights from ariel model."""
    weights = {}
    for i, conn in enumerate(fiu_model.neuralnetwork.getConnections()):
        src = conn.getSource().getName()
        dst = conn.getTarget().getName()
        conn_type = conn.connType
        weight = conn.getTestWeight()
        key = (src, dst, str(conn_type))
        weights[key] = weight
    return weights

def extract_torch_neuron_states(twc, neuron_names_by_layer):
    """Extract neuron states from PyTorch TWC."""
    states = {}
    if twc._state is None:
        return states
    
    # Map layer states to neuron names
    layer_map = {
        'in': neuron_names_by_layer['input'],
        'hid': neuron_names_by_layer['hidden'],
        'out': neuron_names_by_layer['output'],
    }
    
    for layer_name, names in layer_map.items():
        if layer_name in twc._state:
            E, O = twc._state[layer_name]
            for idx, name in enumerate(names):
                states[name] = {
                    'internal': E[0, idx].item() if E.dim() > 1 else E[idx].item(),
                    'output': O[0, idx].item() if O.dim() > 1 else O[idx].item(),
                }
    
    # Extract thresholds and decays from layers
    for layer_name, names in layer_map.items():
        if layer_name == 'in':
            layer = twc.in_layer
        elif layer_name == 'hid':
            layer = twc.hid_layer
        elif layer_name == 'out':
            layer = twc.out_layer
        else:
            continue
            
        for idx, name in enumerate(names):
            if name not in states:
                states[name] = {}
            states[name]['threshold'] = layer.threshold[idx].item()
            states[name]['decay'] = layer.decay[idx].item()
    
    return states

def sync_weights_from_ariel_to_torch(fiu_model, twc, neuron_names_by_layer):
    """Synchronize weights from ariel model to PyTorch TWC."""
    ariel_weights = extract_ariel_connection_weights(fiu_model)
    
    # Get neuron index mappings
    input_idx = {name: i for i, name in enumerate(neuron_names_by_layer['input'])}
    hidden_idx = {name: i for i, name in enumerate(neuron_names_by_layer['hidden'])}
    output_idx = {name: i for i, name in enumerate(neuron_names_by_layer['output'])}
    
    # Softplus inverse to set weights correctly
    softplus_inv_one = math.log(math.e - 1.0)
    
    with torch.no_grad():
        # Sync in2hid connections
        for (src, dst, conn_type_str), weight in ariel_weights.items():
            if src in input_idx and dst in hidden_idx:
                src_i = input_idx[src]
                dst_i = hidden_idx[dst]
                
                if 'ChemEx' in conn_type_str or 'ChemEx' == conn_type_str:
                    # EX connection (should go to in2hid_EX if it exists, but TWC only has in2hid_IN)
                    # Check if this connection should be EX or IN based on the mask
                    # For now, we'll skip EX connections from input to hidden as TWC doesn't have in2hid_EX
                    pass
                elif 'ChemIn' in conn_type_str or 'ChemIn' == conn_type_str:
                    # IN connection
                    if hasattr(twc, 'in2hid_IN') and twc.in2hid_IN.w_mask[src_i, dst_i] > 0:
                        # Set weight to produce desired value after softplus
                        # Since softplus(w) * mask = weight, we need w such that softplus(w) = weight
                        if weight > 0:
                            w_inv = math.log(math.exp(weight) - 1.0) if weight < 20 else weight
                            twc.in2hid_IN.w[src_i, dst_i] = w_inv
        
        # Sync hidden layer connections
        for (src, dst, conn_type_str), weight in ariel_weights.items():
            if src in hidden_idx and dst in hidden_idx:
                src_i = hidden_idx[src]
                dst_i = hidden_idx[dst]
                
                if 'ChemEx' in conn_type_str or 'ChemEx' == conn_type_str:
                    if hasattr(twc, 'hid_EX') and twc.hid_EX.w_mask[src_i, dst_i] > 0:
                        if weight > 0:
                            w_inv = math.log(math.exp(weight) - 1.0) if weight < 20 else weight
                            twc.hid_EX.w[src_i, dst_i] = w_inv
                elif 'ChemIn' in conn_type_str or 'ChemIn' == conn_type_str:
                    if hasattr(twc, 'hid_IN') and twc.hid_IN.w_mask[src_i, dst_i] > 0:
                        if weight > 0:
                            w_inv = math.log(math.exp(weight) - 1.0) if weight < 20 else weight
                            twc.hid_IN.w[src_i, dst_i] = w_inv
        
        # Sync hid2out connections
        for (src, dst, conn_type_str), weight in ariel_weights.items():
            if src in hidden_idx and dst in output_idx:
                src_i = hidden_idx[src]
                dst_i = output_idx[dst]
                
                if 'ChemEx' in conn_type_str or 'ChemEx' == conn_type_str:
                    if hasattr(twc, 'hid2out') and twc.hid2out.w_mask[src_i, dst_i] > 0:
                        if weight > 0:
                            w_inv = math.log(math.exp(weight) - 1.0) if weight < 20 else weight
                            twc.hid2out.w[src_i, dst_i] = w_inv
        
        # Sync gap junction connections
        # GJ edges are: PLM->PVC (input[1] -> hidden[2]), AVM->AVD (input[2] -> hidden[1])
        # Based on twc_builder.py: gj_edges = [[1, 2], [2, 1]] where first row is src, second is dst
        if hasattr(twc, 'in2hid_GJ'):
            gj_src = twc.in2hid_GJ.gj_idx[0]  # (E,) tensor of source indices
            gj_dst = twc.in2hid_GJ.gj_idx[1]  # (E,) tensor of destination indices
            for (src, dst, conn_type_str), weight in ariel_weights.items():
                if 'GJ' in conn_type_str or 'AGJ' in conn_type_str or 'SGJ' in conn_type_str:
                    if src in input_idx and dst in hidden_idx:
                        src_i = input_idx[src]
                        dst_i = hidden_idx[dst]
                        # Find matching edge
                        for edge_idx in range(gj_src.shape[0]):
                            if gj_src[edge_idx].item() == src_i and gj_dst[edge_idx].item() == dst_i:
                                # Compute softplus inverse: if softplus(x) = w, then x = log(exp(w) - 1)
                                if weight > 0:
                                    w_inv = math.log(math.exp(weight) - 1.0) if weight < 20 else weight
                                    twc.in2hid_GJ.gj_w[edge_idx] = w_inv
                                break

def sync_weights_from_torch_to_ariel(twc, fiu_model, neuron_names_by_layer):
    """Synchronize weights from PyTorch TWC to ariel model (reverse direction)."""
    # Build index maps
    input_idx = {name: i for i, name in enumerate(neuron_names_by_layer['input'])}
    hidden_idx = {name: i for i, name in enumerate(neuron_names_by_layer['hidden'])}
    output_idx = {name: i for i, name in enumerate(neuron_names_by_layer['output'])}

    # Convenience: softplus for positive weights
    sp = torch.nn.functional.softplus

    # Extract GJ index arrays if present
    gj_src = twc.in2hid_GJ.gj_idx[0] if hasattr(twc, 'in2hid_GJ') else None
    gj_dst = twc.in2hid_GJ.gj_idx[1] if hasattr(twc, 'in2hid_GJ') else None

    # Iterate Ariel connections and set test weights from TWC effective weights
    for conn in fiu_model.neuralnetwork.getConnections():
        src = conn.getSource().getName()
        dst = conn.getTarget().getName()
        ctype = conn.connType

        # ChemIn
        if 'ChemIn' in str(ctype):
            if src in input_idx and dst in hidden_idx and hasattr(twc, 'in2hid_IN'):
                si, di = input_idx[src], hidden_idx[dst]
                mask_val = twc.in2hid_IN.w_mask[si, di].item() if hasattr(twc.in2hid_IN, 'w_mask') else 1.0
                val = sp(twc.in2hid_IN.w[si, di]).item() * mask_val
                conn.setTestWeight(max(0.0, float(val)))
            elif src in hidden_idx and dst in hidden_idx and hasattr(twc, 'hid_IN'):
                si, di = hidden_idx[src], hidden_idx[dst]
                mask_val = twc.hid_IN.w_mask[si, di].item() if hasattr(twc.hid_IN, 'w_mask') else 1.0
                val = sp(twc.hid_IN.w[si, di]).item() * mask_val
                conn.setTestWeight(max(0.0, float(val)))

        # ChemEx
        elif 'ChemEx' in str(ctype):
            if src in hidden_idx and dst in hidden_idx and hasattr(twc, 'hid_EX'):
                si, di = hidden_idx[src], hidden_idx[dst]
                mask_val = twc.hid_EX.w_mask[si, di].item() if hasattr(twc.hid_EX, 'w_mask') else 1.0
                val = sp(twc.hid_EX.w[si, di]).item() * mask_val
                conn.setTestWeight(max(0.0, float(val)))
            elif src in hidden_idx and dst in output_idx and hasattr(twc, 'hid2out'):
                si, di = hidden_idx[src], output_idx[dst]
                mask_val = twc.hid2out.w_mask[si, di].item() if hasattr(twc.hid2out, 'w_mask') else 1.0
                val = sp(twc.hid2out.w[si, di]).item() * mask_val
                conn.setTestWeight(max(0.0, float(val)))

        # Gap junctions: AGJ / SGJ
        else:
            if gj_src is None or gj_dst is None:
                continue
            if src in input_idx and dst in hidden_idx:
                si, di = input_idx[src], hidden_idx[dst]
                # Find the matching edge index
                for e in range(gj_src.shape[0]):
                    if gj_src[e].item() == si and gj_dst[e].item() == di:
                        val = sp(twc.in2hid_GJ.gj_w[e]).item()
                        conn.setTestWeight(max(0.0, float(val)))
                        break

def sync_neuron_params_from_ariel_to_torch(fiu_model, twc, neuron_names_by_layer):
    """Synchronize neuron thresholds and decay factors."""
    ariel_states = extract_ariel_neuron_states(fiu_model)
    
    with torch.no_grad():
        # Sync input layer
        for idx, name in enumerate(neuron_names_by_layer['input']):
            if name in ariel_states:
                twc.in_layer.threshold[idx] = ariel_states[name]['threshold']
                twc.in_layer.decay[idx] = ariel_states[name]['decay']
        
        # Sync hidden layer
        for idx, name in enumerate(neuron_names_by_layer['hidden']):
            if name in ariel_states:
                twc.hid_layer.threshold[idx] = ariel_states[name]['threshold']
                twc.hid_layer.decay[idx] = ariel_states[name]['decay']
        
        # Sync output layer
        for idx, name in enumerate(neuron_names_by_layer['output']):
            if name in ariel_states:
                twc.out_layer.threshold[idx] = ariel_states[name]['threshold']
                twc.out_layer.decay[idx] = ariel_states[name]['decay']

def sync_neuron_params_from_torch_to_ariel(twc, fiu_model, neuron_names_by_layer):
    """Synchronize neuron thresholds and decay factors from TWC to Ariel."""
    # Helper to set per layer
    def set_layer_params(layer, names, set_th_fn, set_df_fn):
        for idx, name in enumerate(names):
            th = float(layer.threshold[idx].item())
            df = float(layer.decay[idx].item())
            set_th_fn(name, th)
            set_df_fn(name, df)

    nn = fiu_model.neuralnetwork
    set_layer_params(twc.in_layer, neuron_names_by_layer['input'], nn.setNeuronTestThresholdOfName, nn.setNeuronTestDecayFactorOfName)
    set_layer_params(twc.hid_layer, neuron_names_by_layer['hidden'], nn.setNeuronTestThresholdOfName, nn.setNeuronTestDecayFactorOfName)
    set_layer_params(twc.out_layer, neuron_names_by_layer['output'], nn.setNeuronTestThresholdOfName, nn.setNeuronTestDecayFactorOfName)

def get_neuron_names_from_json():
    """Get neuron names organized by layer from TWC JSON."""
    json_path = Path(__file__).parent.parent / "src" / "twc" / "TWC_fiu.json"
    with open(json_path, "r") as f:
        net_data = json.load(f)
    return net_data["groups"]

def convert_ariel_output_to_action(fiu_output):
    """Convert ariel model output (scalar) to match action decoder format."""
    # The ariel model returns a scalar from OUT1 interface
    # This is already the action value, but we need to ensure it's in the right format
    return fiu_output

def compare_states(ariel_states, torch_states, tolerance=1e-5):
    """Compare neuron states between implementations."""
    differences = {}
    all_match = True
    
    for name in set(ariel_states.keys()) | set(torch_states.keys()):
        if name not in ariel_states:
            differences[name] = {'error': 'Missing in ariel'}
            all_match = False
            continue
        if name not in torch_states:
            differences[name] = {'error': 'Missing in torch'}
            all_match = False
            continue
        
        a_state = ariel_states[name]
        t_state = torch_states[name]
        
        diff = {}
        for key in ['internal', 'output', 'threshold', 'decay']:
            if key in a_state and key in t_state:
                a_val = a_state[key]
                t_val = t_state[key]
                error = abs(a_val - t_val)
                diff[key] = {
                    'ariel': a_val,
                    'torch': t_val,
                    'error': error,
                    'match': error < tolerance
                }
                if error >= tolerance:
                    all_match = False
        
        if diff:
            differences[name] = diff
    
    return differences, all_match

def main():
    env = gym.make(ENV)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    torch.manual_seed(SEED)

    out_dir = os.path.join('out/tests/twc_validation')
    os.makedirs(out_dir, exist_ok=True)

    # Get neuron name mappings
    neuron_names = get_neuron_names_from_json()
    
    # Build TWC for MCC with logging enabled
    twc = build_twc(
        obs_encoder=mcc_obs_encoder,
        action_decoder=twc_out_2_mcc_action,
        internal_steps=1,  # Match ariel's single step
        log_stats=True,
    )
    twc.eval()

    # Load ariel model
    xml_path = os.path.join(Path(__file__).parent, 'TWFiuriBaseFIU.xml')
    fiu_twc = FiuModel('FIU')
    fiu_twc.loadFromFile(xml_path)
    fiu_twc.Reset()
    
    print("Synchronizing weights and parameters (Torch -> Ariel)...")
    sync_weights_from_torch_to_ariel(twc, fiu_twc, neuron_names)
    sync_neuron_params_from_torch_to_ariel(twc, fiu_twc, neuron_names)
    
    # Reset both models
    #fiu_twc.Reset()
    #twc.reset()
    
    # Generate test observations
    num_tests = 1000
    test_observations = []
    env.reset()
    for i in range(num_tests):
        obs = env.observation_space.sample()
        test_observations.append(obs)
    
    # Storage for comparisons
    output_differences = []
    state_differences_all = []
    ariel_outputs = []
    torch_outputs = []

    # Per-neuron mismatch tracking and error stats
    # Structure: { neuron_name: { key: {"count": int, "errors": [float]} } }
    per_neuron_stats = {}
    
    print(f"\nRunning {num_tests} test cases...")
    print("=" * 80)
    
    for test_idx, obs in enumerate(test_observations):
        print(f"\nTest {test_idx + 1}/{num_tests}: obs={obs}")

        if test_idx % 10 == 0:
            fiu_twc.Reset()  
            twc.reset()
        
        ariel_output = fiu_twc.Update(obs, mode=None, doLog=False)
        ariel_states = extract_ariel_neuron_states(fiu_twc)
        ariel_outputs.append(ariel_output)
        
        # Run torch model
        obs_tensor = torch.tensor([obs], dtype=torch.float32)
        with torch.no_grad():
            torch_output = twc(obs_tensor)
        
        torch_states = extract_torch_neuron_states(twc, neuron_names)
        torch_outputs.append(torch_output.squeeze().item())
        
        # Compare outputs
        output_diff = abs(ariel_output - torch_outputs[-1])
        output_differences.append(output_diff)
        print(f"  Output: ariel={ariel_output:.6f}, torch={torch_outputs[-1]:.6f}, diff={output_diff:.6f}")
        
        # Compare states
        state_diffs, states_match = compare_states(ariel_states, torch_states, tolerance=1e-4)
        state_differences_all.append(state_diffs)

        # Aggregate per-neuron mismatches and errors
        for name, diffs in state_diffs.items():
            if 'error' in diffs:
                continue
            if name not in per_neuron_stats:
                per_neuron_stats[name] = {
                    'internal': {"count": 0, "errors": []},
                    'output':   {"count": 0, "errors": []},
                    'threshold':{"count": 0, "errors": []},
                    'decay':    {"count": 0, "errors": []},
                }
            for key, val in diffs.items():
                if not val['match']:
                    per_neuron_stats[name][key]["count"] += 1
                    per_neuron_stats[name][key]["errors"].append(val['error'])
        
        if not states_match:
            print(f"  ⚠ State differences detected:")
            for name, diffs in state_diffs.items():
                if 'error' in diffs:
                    print(f"    {name}: {diffs['error']}")
                else:
                    for key, val in diffs.items():
                        if not val['match']:
                            print(f"    {name}.{key}: ariel={val['ariel']:.6f}, torch={val['torch']:.6f}, error={val['error']:.6f}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    output_diffs_array = np.array(output_differences)
    print(f"Output differences:")
    print(f"  Mean: {output_diffs_array.mean():.6f}")
    print(f"  Std:  {output_diffs_array.std():.6f}")
    print(f"  Max:  {output_diffs_array.max():.6f}")
    print(f"  Min:  {output_diffs_array.min():.6f}")
    
    # Count state mismatches
    total_state_checks = 0
    state_mismatches = 0
    for state_diffs in state_differences_all:
        for name, diffs in state_diffs.items():
            if 'error' not in diffs:
                for key, val in diffs.items():
                    total_state_checks += 1
                    if not val['match']:
                        state_mismatches += 1
    
    if total_state_checks > 0:
        match_rate = (1 - state_mismatches / total_state_checks) * 100
        print(f"\nState comparison:")
        print(f"  Total checks: {total_state_checks}")
        print(f"  Mismatches: {state_mismatches}")
        print(f"  Match rate: {match_rate:.2f}%")

    # Compute per-neuron aggregated statistics
    def _agg_errs(errs):
        if not errs:
            return 0, 0.0, 0.0
        arr = np.array(errs, dtype=np.float32)
        return len(errs), float(arr.mean()), float(arr.max())

    # Flatten a simple summary for console: top neurons by total mismatches
    per_neuron_totals = []
    for neuron, keys in per_neuron_stats.items():
        total = sum(v["count"] for v in keys.values())
        per_neuron_totals.append((neuron, total))
    per_neuron_totals.sort(key=lambda x: x[1], reverse=True)

    if per_neuron_totals:
        print("\nTop per-neuron mismatches:")
        for neuron, total in per_neuron_totals[:10]:
            print(f"  {neuron}: {total}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Output comparison
    ax = axes[0, 0]
    ax.scatter(ariel_outputs, torch_outputs, alpha=0.6)
    ax.plot([min(ariel_outputs + torch_outputs), max(ariel_outputs + torch_outputs)],
            [min(ariel_outputs + torch_outputs), max(ariel_outputs + torch_outputs)],
            'r--', label='y=x')
    ax.set_xlabel('Ariel Output')
    ax.set_ylabel('Torch Output')
    ax.set_title('Output Comparison')
    ax.legend()
    ax.grid(True)
    
    # Output differences
    ax = axes[0, 1]
    ax.plot(output_differences, 'o-')
    ax.set_xlabel('Test Case')
    ax.set_ylabel('Absolute Difference')
    ax.set_title('Output Differences Over Tests')
    ax.grid(True)
    
    # Output distribution
    ax = axes[1, 0]
    ax.hist(ariel_outputs, alpha=0.5, label='Ariel', bins=15)
    ax.hist(torch_outputs, alpha=0.5, label='Torch', bins=15)
    ax.set_xlabel('Output Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Output Distribution')
    ax.legend()
    ax.grid(True)
    
    # Error distribution
    ax = axes[1, 1]
    ax.hist(output_differences, bins=15, edgecolor='black')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'twc_comparison.png'), dpi=150)
    print(f"\nVisualization saved to {os.path.join(out_dir, 'twc_comparison.png')}")
    
    # Save detailed comparison to file
    with open(os.path.join(out_dir, 'detailed_comparison.txt'), 'w') as f:
        f.write("TWC Implementation Comparison\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of test cases: {num_tests}\n\n")
        f.write("Output Statistics:\n")
        f.write(f"  Mean difference: {output_diffs_array.mean():.6f}\n")
        f.write(f"  Std difference:  {output_diffs_array.std():.6f}\n")
        f.write(f"  Max difference:  {output_diffs_array.max():.6f}\n")
        f.write(f"  Min difference:  {output_diffs_array.min():.6f}\n\n")
        
        if total_state_checks > 0:
            f.write(f"State Comparison:\n")
            f.write(f"  Total checks: {total_state_checks}\n")
            f.write(f"  Mismatches: {state_mismatches}\n")
            f.write(f"  Match rate: {match_rate:.2f}%\n\n")

        # Per-neuron mismatch summary
        f.write("Per-neuron mismatch summary:\n")
        f.write("-" * 80 + "\n")
        # Order by total mismatches
        for neuron, total in per_neuron_totals:
            f.write(f"{neuron}: total_mismatches={total}\n")
            for key in ("internal", "output", "threshold", "decay"):
                cnt, mean_err, max_err = _agg_errs(per_neuron_stats.get(neuron, {}).get(key, {}).get("errors", []))
                if cnt > 0:
                    f.write(f"  - {key}: count={per_neuron_stats[neuron][key]['count']}, mean_err={mean_err:.6f}, max_err={max_err:.6f}\n")
        f.write("\n")
        
        f.write("\nPer-test details:\n")
        f.write("-" * 80 + "\n")
        for test_idx, (obs, a_out, t_out, state_diffs) in enumerate(zip(
            test_observations, ariel_outputs, torch_outputs, state_differences_all)):
            f.write(f"\nTest {test_idx + 1}:\n")
            f.write(f"  Observation: {obs}\n")
            f.write(f"  Ariel output: {a_out:.6f}\n")
            f.write(f"  Torch output: {t_out:.6f}\n")
            f.write(f"  Difference: {abs(a_out - t_out):.6f}\n")
            if state_diffs:
                f.write(f"  State differences:\n")
                for name, diffs in state_diffs.items():
                    if 'error' in diffs:
                        f.write(f"    {name}: {diffs['error']}\n")
                    else:
                        for key, val in diffs.items():
                            if not val['match']:
                                f.write(f"    {name}.{key}: error={val['error']:.6f}\n")
    
    print(f"Detailed comparison saved to {os.path.join(out_dir, 'detailed_comparison.txt')}")

     # =============================
    # Stateful multi-step evaluation
    # =============================
    def align_states(fiu_model, twc_model, align_steps: int = 3):
        neutral = np.array([POS_VALLEY_VAL, VEL_VALLEY_VAL], dtype=np.float32)
        obs_tensor = torch.tensor([neutral], dtype=torch.float32)
        for _ in range(align_steps):
            # Run both without resets to drive to a common quiescent state
            fiu_model.Update(neutral, mode=None, doLog=False)
            with torch.no_grad():
                twc_model(obs_tensor)
    def gen_pulsed_sequence(seq_len: int, pulse_every: int = 5):
        seq = []
        for t in range(seq_len):
            if t % pulse_every == 0:
                # Strong activating sample: toggle around both interfaces
                pos = np.random.choice([POS_MIN, POS_MAX])
                vel = np.random.choice([-VEL_MAX, VEL_MAX])
            else:
                # Near valley to exercise both branches and avoid full quiescence
                pos = float(POS_VALLEY_VAL + np.random.uniform(-0.1, 0.1))
                vel = float(VEL_VALLEY_VAL + np.random.uniform(-0.03, 0.03))
                pos = max(POS_MIN, min(POS_MAX, pos))
                vel = max(-VEL_MAX, min(VEL_MAX, vel))
            seq.append(np.array([pos, vel], dtype=np.float32))
        return seq
    def run_stateful_eval(num_sequences: int = 25, seq_len: int = 30, burn_in: int = 3, pulse_every: int = 5, eps: float = 1e-3):
        active_diffs = []
        all_diffs = []
        per_neuron_stats_active = {}
        for s in range(num_sequences):
            # Reset models at the start of each sequence
            fiu_twc.Reset()
            twc.reset()
            # Align to a common operating point
            align_states(fiu_twc, twc, align_steps=burn_in)
            seq = gen_pulsed_sequence(seq_len, pulse_every=pulse_every)
            for obs in seq:
                # Step without resets (stateful)
                ar_out = fiu_twc.Update(obs, mode=None, doLog=False)
                with torch.no_grad():
                    t_out = twc(torch.tensor([obs], dtype=torch.float32)).squeeze().item()
                diff = abs(ar_out - t_out)
                all_diffs.append(diff)
                # Active-only error tracking
                if (abs(ar_out) > eps) or (abs(t_out) > eps):
                    active_diffs.append(diff)
                    # Per-neuron mismatch on active steps
                    a_states = extract_ariel_neuron_states(fiu_twc)
                    t_states = extract_torch_neuron_states(twc, neuron_names)
                    state_diffs, _ = compare_states(a_states, t_states, tolerance=1e-4)
                    for name, diffs in state_diffs.items():
                        if 'error' in diffs:
                            continue
                        if name not in per_neuron_stats_active:
                            per_neuron_stats_active[name] = {
                                'internal': {"count": 0, "errors": []},
                                'output':   {"count": 0, "errors": []},
                                'threshold':{"count": 0, "errors": []},
                                'decay':    {"count": 0, "errors": []},
                            }
                        for key, val in diffs.items():
                            if not val['match']:
                                per_neuron_stats_active[name][key]['count'] += 1
                                per_neuron_stats_active[name][key]['errors'].append(val['error'])
        all_diffs_arr = np.array(all_diffs, dtype=np.float32)
        active_diffs_arr = np.array(active_diffs, dtype=np.float32) if active_diffs else np.array([], dtype=np.float32)
        def _agg_errs(errs):
            if not errs:
                return 0, 0.0, 0.0
            arr = np.array(errs, dtype=np.float32)
            return len(errs), float(arr.mean()), float(arr.max())
        per_neuron_totals_active = []
        for neuron, keys in per_neuron_stats_active.items():
            total = sum(v['count'] for v in keys.values())
            per_neuron_totals_active.append((neuron, total))
        per_neuron_totals_active.sort(key=lambda x: x[1], reverse=True)
        # Save stateful summary
        stateful_path = os.path.join(out_dir, 'stateful_summary.txt')
        with open(stateful_path, 'w') as f:
            f.write("TWC Stateful Evaluation (multi-step)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Sequences: {num_sequences}, Length: {seq_len}, Burn-in: {burn_in}, Pulse-every: {pulse_every}\n")
            f.write(f"Total steps: {num_sequences*seq_len}\n")
            f.write(f"Active steps: {active_diffs_arr.size} (|action|>{eps})\n\n")
            f.write("Action error (all steps):\n")
            f.write(f"  Mean: {all_diffs_arr.mean() if all_diffs_arr.size else 0.0:.6f}\n")
            f.write(f"  Std:  {all_diffs_arr.std()  if all_diffs_arr.size else 0.0:.6f}\n")
            f.write(f"  Max:  {all_diffs_arr.max()  if all_diffs_arr.size else 0.0:.6f}\n")
            f.write(f"  Min:  {all_diffs_arr.min()  if all_diffs_arr.size else 0.0:.6f}\n\n")
            f.write("Action error (active-only):\n")
            f.write(f"  Mean: {active_diffs_arr.mean() if active_diffs_arr.size else 0.0:.6f}\n")
            f.write(f"  Std:  {active_diffs_arr.std()  if active_diffs_arr.size else 0.0:.6f}\n")
            f.write(f"  Max:  {active_diffs_arr.max()  if active_diffs_arr.size else 0.0:.6f}\n")
            f.write(f"  Min:  {active_diffs_arr.min()  if active_diffs_arr.size else 0.0:.6f}\n\n")
            f.write("Per-neuron mismatch summary (active-only):\n")
            f.write("-" * 80 + "\n")
            for neuron, total in per_neuron_totals_active:
                f.write(f"{neuron}: total_mismatches={total}\n")
                for key in ("internal", "output", "threshold", "decay"):
                    cnt, mean_err, max_err = _agg_errs(per_neuron_stats_active.get(neuron, {}).get(key, {}).get("errors", []))
                    if cnt > 0:
                        f.write(f"  - {key}: count={per_neuron_stats_active[neuron][key]['count']}, mean_err={mean_err:.6f}, max_err={max_err:.6f}\n")
        print(f"Stateful summary saved to {stateful_path}")
    # Run stateful evaluation to validate multi-step parity
    run_stateful_eval(num_sequences=25, seq_len=30, burn_in=3, pulse_every=5, eps=1e-3)
    
if __name__ == "__main__":
    main()
