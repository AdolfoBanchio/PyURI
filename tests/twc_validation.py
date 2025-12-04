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
from twc.twc_builder import build_twc, TWC
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
        # Si el estado es None, inicialízalo para obtener los estados E=0, O=0
        batch_size = 1
        device = next(twc.parameters()).device
        dtype = next(twc.parameters()).dtype
        twc._state = twc._make_state(batch_size, device, dtype)

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

def _agg_errs(errs):
    if not errs:
        return 0, 0.0, 0.0
    arr = np.array(errs, dtype=np.float32)
    return len(errs), float(arr.mean()), float(arr.max())

def save_comparison(out_dir, 
                    ariel_outputs, 
                    torch_v1_outputs, torch_v2_outputs, # <--- 
                    output_diffs_v1_array, output_diffs_v2_array, # <--- 
                    state_differences_all,
                    neuron_names,
                    twc, 
                    fiu_twc,
                    num_tests,
                    total_state_checks,
                    state_mismatches,
                    match_rate,
                    per_neuron_stats,
                    per_neuron_totals,
                    test_observations):
    
    with open(os.path.join(out_dir, 'detailed_comparison.txt'), 'w') as f:
            # Save each model parameters for reference into the same txt file
            f.write("Model Parameters\n")
            f.write("=" * 80 + "\n\n")
            # Torch TWC parameters
            f.write("Torch TWC parameters:\n")
            f.write("-" * 80 + "\n")
            # Layer params (threshold, decay)
            f.write("Layers (threshold, decay):\n")
            for layer_name, layer_module, names in (
                ("input", twc.in_layer, neuron_names["input"]),
                ("hidden", twc.hid_layer, neuron_names["hidden"]),
                ("output", twc.out_layer, neuron_names["output"]),
            ):
                f.write(f"  {layer_name}:\n")
                layer_th = layer_module.threshold
                layer_dc = layer_module.decay
                for i, n in enumerate(names):
                    th = float(layer_th[i].item())
                    dc = float(layer_dc[i].item())
                    f.write(f"    {n}: threshold={th:.6f}, decay={dc:.6f}\n")
            f.write("\n")

            # Connection params (effective weights after softplus and mask)
            sp = torch.nn.functional.softplus
            f.write("Connections (effective weights):\n")
            # in2hid_IN
            f.write("  in2hid_IN (ChemIn):\n")
            for i, src in enumerate(neuron_names["input"]):
                for j, dst in enumerate(neuron_names["hidden"]):
                    mask_val = float(twc.in2hid_IN.w_mask[i, j].item()) if hasattr(twc.in2hid_IN, "w_mask") else 1.0
                    if mask_val > 0.0:
                        w_eff = float(sp(twc.in2hid_IN.w[i, j]).item() * mask_val)
                        f.write(f"    {src} -> {dst}: {w_eff:.6f}\n")
            # hid_IN
            f.write("  hid_IN (ChemIn):\n")
            for i, src in enumerate(neuron_names["hidden"]):
                for j, dst in enumerate(neuron_names["hidden"]):
                    mask_val = float(twc.hid_IN.w_mask[i, j].item()) if hasattr(twc.hid_IN, "w_mask") else 1.0
                    if mask_val > 0.0:
                        w_eff = float(sp(twc.hid_IN.w[i, j]).item() * mask_val)
                        f.write(f"    {src} -> {dst}: {w_eff:.6f}\n")
            # hid_EX
            f.write("  hid_EX (ChemEx):\n")
            for i, src in enumerate(neuron_names["hidden"]):
                for j, dst in enumerate(neuron_names["hidden"]):
                    mask_val = float(twc.hid_EX.w_mask[i, j].item()) if hasattr(twc.hid_EX, "w_mask") else 1.0
                    if mask_val > 0.0:
                        w_eff = float(sp(twc.hid_EX.w[i, j]).item() * mask_val)
                        f.write(f"    {src} -> {dst}: {w_eff:.6f}\n")
            # hid2out
            f.write("  hid2out_EX (ChemEx):\n")
            for i, src in enumerate(neuron_names["hidden"]):
                for j, dst in enumerate(neuron_names["output"]):
                    mask_val = float(twc.hid2out.w_mask[i, j].item()) if hasattr(twc.hid2out, "w_mask") else 1.0
                    if mask_val > 0.0:
                        w_eff = float(sp(twc.hid2out.w[i, j]).item() * mask_val)
                        f.write(f"    {src} -> {dst}: {w_eff:.6f}\n")
            # Gap junctions
            if hasattr(twc, "in2hid_GJ") and hasattr(twc.in2hid_GJ, "gj_idx"):
                f.write("  in2hid_GJ (GJ):\n")
                gj_src = twc.in2hid_GJ.gj_idx[0]
                gj_dst = twc.in2hid_GJ.gj_idx[1]
                for e in range(gj_src.shape[0]):
                    si = int(gj_src[e].item())
                    di = int(gj_dst[e].item())
                    src = neuron_names["input"][si]
                    dst = neuron_names["hidden"][di]
                    w_eff = float(sp(twc.in2hid_GJ.gj_w[e]).item())
                    f.write(f"    {src} -> {dst}: {w_eff:.6f}\n")
            f.write("\n")

            # Ariel FIU parameters
            f.write("Ariel FIU parameters:\n")
            f.write("-" * 80 + "\n")
            # Neuron params
            f.write("Neurons (threshold, decay):\n")
            for n in fiu_twc.neuralnetwork.getNeuronNames():
                neuron = fiu_twc.getNeuron(n)
                f.write(
                    f"  {n}: threshold={neuron.getTestThreshold():.6f}, decay={neuron.getTestDecayFactor():.6f}\n"
                )
            # Connection params
            f.write("\nConnections (weights):\n")
            for conn in fiu_twc.neuralnetwork.getConnections():
                src = conn.getSource().getName()
                dst = conn.getTarget().getName()
                ctype = str(conn.connType)
                w = conn.getTestWeight()
                f.write(f"  {src} -> {dst} [{ctype}]: {w:.6f}\n")
            f.write("\n")
            
            # Save detailed summary
            f.write("TWC Implementation Comparison\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Number of test cases: {num_tests}\n\n")
            
            f.write("Output Statistics (Ariel vs V1 - Precise):\n")
            f.write(f"  Mean difference: {output_diffs_v1_array.mean():.6f}\n")
            f.write(f"  Std difference:  {output_diffs_v1_array.std():.6f}\n")
            f.write(f"  Max difference:  {output_diffs_v1_array.max():.6f}\n")
            f.write(f"  Min difference:  {output_diffs_v1_array.min():.6f}\n\n")
            
            f.write("Output Statistics (Ariel vs V2 - Differentiable):\n")
            f.write(f"  Mean difference: {output_diffs_v2_array.mean():.6f}\n")
            f.write(f"  Std difference:  {output_diffs_v2_array.std():.6f}\n")
            f.write(f"  Max difference:  {output_diffs_v2_array.max():.6f}\n")
            f.write(f"  Min difference:  {output_diffs_v2_array.min():.6f}\n\n")
            
            if total_state_checks > 0:
                f.write(f"State Comparison (Ariel vs V1):\n") # <--- 
                f.write(f"  Total checks: {total_state_checks}\n")
                f.write(f"  Mismatches: {state_mismatches}\n")
                f.write(f"  Match rate: {match_rate:.2f}%\n\n")

            f.write("-" * 80 + "\n")
            # Order by total mismatches
            f.write("Per-neuron mismatch summary (Ariel vs V1):\n")
            f.write("-" * 80 + "\n")
            for neuron, total in per_neuron_totals:
                f.write(f"{neuron}: total_mismatches={total}\n")
                for key in ("internal", "output", "threshold", "decay"):
                    cnt, mean_err, max_err = _agg_errs(per_neuron_stats.get(neuron, {}).get(key, {}).get("errors", []))
                    if cnt > 0:
                        f.write(f"  - {key}: count={cnt}, mean_err={mean_err:.6f}, max_err={max_err:.6f}\n")
            f.write("\n")
            
            f.write("\nPer-test details:\n")
            f.write("-" * 80 + "\n")
            for test_idx, (obs, a_out, t_v1_out, t_v2_out, state_diffs) in enumerate(zip(
                test_observations, ariel_outputs, torch_v1_outputs, torch_v2_outputs, state_differences_all)): # <--- 
                f.write(f"\nTest {test_idx + 1}:\n")
                f.write(f"  Observation: {obs}\n")
                f.write(f"  Ariel output:     {a_out:.6f}\n")
                f.write(f"  Torch V1 output:  {t_v1_out:.6f} (Diff: {abs(a_out - t_v1_out):.6f})\n") # <--- 
                f.write(f"  Torch V2 output:  {t_v2_out:.6f} (Diff: {abs(a_out - t_v2_out):.6f})\n") # <--- 

                has_diff = any(not v['match'] for d in state_diffs.values() if 'error' not in d for k, v in d.items())
                if has_diff:
                    f.write(f"  State differences (Ariel vs V1):\n")
                    for name, diffs in state_diffs.items():
                        if 'error' in diffs:
                            f.write(f"    {name}: {diffs['error']}\n")
                        else:
                            for key, val in diffs.items():
                                if not val['match']:
                                    f.write(f"    {name}.{key}: error={val['error']:.6f}\n")
        
    print(f"Detailed comparison saved to {os.path.join(out_dir, 'detailed_comparison.txt')}")

def main():
    env = gym.make(ENV)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = os.path.join('out/tests/twc_validation')
    os.makedirs(out_dir, exist_ok=True)

    # Get neuron name mappings
    neuron_names = get_neuron_names_from_json()
    
    print("Building TWC V1 (Precise)...")
    twc_v1 = build_twc(
        obs_encoder=mcc_obs_encoder,
        action_decoder=twc_out_2_mcc_action,
        initial_thresholds=[0.0,0.0,0.0],
        initial_decays=[0.1,0.1,0.1],
        rnd_init=True,
        use_V2=False, # <-- V1
        log_stats=True,
    )
    twc_v1.eval()

    print("Building TWC V2 (Differentiable)...")
    v2_params = {
        'steepness_fire': 1,
        'steepness_gj': 1,
        'steepness_input': 1,
        'input_thresh': 0,
        'leaky_slope': 0.2
    }
    twc_v2 = build_twc(
        obs_encoder=mcc_obs_encoder,
        action_decoder=twc_out_2_mcc_action,
        initial_thresholds=[0.0,0.0,0.0],
        initial_decays=[0.1,0.1,0.1],
        rnd_init=True,
        use_V2=True, # <-- V2
        log_stats=True,
        **{'v2_params': v2_params}
    )
    twc_v2.eval()
    
    print("Copying state_dict from V1 to V2...")
    twc_v2.load_state_dict(twc_v1.state_dict())

    # Load ariel model
    xml_path = os.path.join(Path(__file__).parent, 'TWFiuriBaseFIU.xml')
    fiu_twc = FiuModel('FIU')
    fiu_twc.loadFromFile(xml_path)
    fiu_twc.Reset()
    
    print("Synchronizing weights and parameters (Torch V1 -> Ariel)...")
    sync_weights_from_torch_to_ariel(twc_v1, fiu_twc, neuron_names)
    sync_neuron_params_from_torch_to_ariel(twc_v1, fiu_twc, neuron_names)
    
    print("TWC V1 state_dict:", twc_v1.state_dict())
    print("="*40)
    print("TWC V2 state_dict:", twc_v2.state_dict())
    print("="*40)

    # Generate test observations
    num_tests = 1000
    test_observations = []
    env.reset()
    for i in range(num_tests):
        obs = env.observation_space.sample()
        test_observations.append(obs)
    
    # Storage for comparisons
    output_differences_v1 = []
    output_differences_v2 = []
    state_differences_all = []
    ariel_outputs = []
    torch_v1_outputs = []
    torch_v2_outputs = []
    per_neuron_stats = {}
    
    print(f"\nRunning {num_tests} test cases...")
    print("=" * 80)
    
    for test_idx, obs in enumerate(test_observations):

        if test_idx % 5 == 0:
            print(f"\n--- Step {test_idx}: Resetting states ---")
            fiu_twc.Reset()  
            twc_v1.reset_internal_only()
            twc_v2.reset_internal_only()

        # 1. Modelo Ariel
        ariel_output = fiu_twc.Update(obs, mode=None, doLog=False)
        ariel_states = extract_ariel_neuron_states(fiu_twc)
        ariel_outputs.append(ariel_output)
        
        # 2. Modelo Torch V1 (Preciso)
        obs_tensor = torch.tensor([obs], dtype=torch.float32)
        with torch.no_grad():
            torch_v1_output = twc_v1(obs_tensor)
        
        torch_v1_states = extract_torch_neuron_states(twc_v1, neuron_names)
        torch_v1_outputs.append(torch_v1_output.squeeze().item())

        # 3. Modelo Torch V2 (Diferenciable)
        with torch.no_grad():
            torch_v2_output = twc_v2(obs_tensor)
        torch_v2_outputs.append(torch_v2_output.squeeze().item())

        # --- MODIFICADO: Comparar ambos ---
        output_diff_v1 = abs(ariel_output - torch_v1_outputs[-1])
        output_diff_v2 = abs(ariel_output - torch_v2_outputs[-1])
        output_differences_v1.append(output_diff_v1)
        output_differences_v2.append(output_diff_v2)

        print(f"\nTest {test_idx + 1}/{num_tests}: obs={obs}")
        print(f"  Ariel Output: {ariel_output:.6f}")
        print(f"  V1 Output:    {torch_v1_outputs[-1]:.6f} (Diff: {output_diff_v1:.6E})")
        print(f"  V2 Output:    {torch_v2_outputs[-1]:.6f} (Diff: {output_diff_v2:.6E})")
        
        # Compare states
        state_diffs, states_match = compare_states(ariel_states, torch_v1_states, tolerance=1e-4)
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
            print(f"State differences detected:")
            for name, diffs in state_diffs.items():
                if 'error' in diffs:
                    print(f"    {name}: {diffs['error']}")
                else:
                    for key, val in diffs.items():
                        if not val['match']:
                            print(f"    {name}.{key}: ariel={val['ariel']}, torch={val['torch']}, error={val['error']}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    output_diffs_v1_array = np.array(output_differences_v1)
    output_diffs_v2_array = np.array(output_differences_v2)
    
    print(f"Output differences (Ariel vs V1 - Precise):")
    print(f"  Mean: {output_diffs_v1_array.mean():.6E}")
    print(f"  Std:  {output_diffs_v1_array.std():.6E}")
    print(f"  Max:  {output_diffs_v1_array.max():.6E}")
    print(f"  Min:  {output_diffs_v1_array.min():.6E}")
    
    print(f"\nOutput differences (Ariel vs V2 - Differentiable):")
    print(f"  Mean: {output_diffs_v2_array.mean():.6E}")
    print(f"  Std:  {output_diffs_v2_array.std():.6E}")
    print(f"  Max:  {output_diffs_v2_array.max():.6E}")
    print(f"  Min:  {output_diffs_v2_array.min():.6E}")
    
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
    
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Ariel (Original) vs Torch V1 (Precise) vs Torch V2 (Differentiable)")
    
    # Output comparison
    ax = axes[0, 0]
    ax.scatter(ariel_outputs, torch_v1_outputs, alpha=0.6, label='Torch V1 (Precise)', s=20)
    ax.scatter(ariel_outputs, torch_v2_outputs, alpha=0.6, label='Torch V2 (Grads)', s=20)
    # Línea y=x
    min_val = min(min(ariel_outputs), min(torch_v1_outputs), min(torch_v2_outputs)) - 0.1
    max_val = max(max(ariel_outputs), max(torch_v1_outputs), max(torch_v2_outputs)) + 0.1
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    ax.set_xlabel('Ariel Output')
    ax.set_ylabel('Torch Output')
    ax.set_title('Output Comparison')
    ax.legend()
    ax.grid(True)
    
    # Output differences
    ax = axes[0, 1]
    ax.plot(output_differences_v1, 'o-', label='V1 (Precise) Diff', markersize=4, alpha=0.7)
    ax.plot(output_differences_v2, 'o-', label='V2 (Grads) Diff', markersize=4, alpha=0.7)
    ax.set_xlabel('Test Case')
    ax.set_ylabel('Absolute Difference')
    ax.set_title('Output Differences Over Tests')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log') # <--- MODIFICADO: Log scale para ver ambas diferencias
    
    # Output distribution
    ax = axes[1, 0]
    ax.hist(ariel_outputs, alpha=0.5, label='Ariel', bins=20)
    ax.hist(torch_v1_outputs, alpha=0.5, label='Torch V1', bins=20, histtype='step', linewidth=2)
    ax.hist(torch_v2_outputs, alpha=0.5, label='Torch V2 (Grads)', bins=20, histtype='step', linewidth=2)
    ax.set_xlabel('Output Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Output Distribution')
    ax.legend()
    ax.grid(True)
    
    # Error distribution
    ax = axes[1, 1]
    ax.hist(output_diffs_v1_array, bins=20, edgecolor='blue', alpha=0.7, label='V1 (Precise) Errors')
    ax.hist(output_diffs_v2_array, bins=20, edgecolor='orange', alpha=0.7, label='V2 (Grads) Errors')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log') # <--- MODIFICADO: Log scale
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para el suptitle
    plt.savefig(os.path.join(out_dir, 'twc_comparison_v1_v2.png'), dpi=150)
    print(f"\nVisualization saved to {os.path.join(out_dir, 'twc_comparison_v1_v2.png')}")
    
    
    # --- MODIFICADO: Guardar comparación para V1 y V2 ---
    save_comparison(
        out_dir,
        ariel_outputs,
        torch_v1_outputs, torch_v2_outputs,
        output_diffs_v1_array, output_diffs_v2_array,
        state_differences_all,
        neuron_names,
        twc_v1, # Usamos V1 como referencia para los parámetros
        fiu_twc,
        num_tests,
        total_state_checks,
        state_mismatches,
        match_rate if total_state_checks > 0 else 100.0,
        per_neuron_stats,
        per_neuron_totals,
        test_observations,
    )
    

    
if __name__ == "__main__":
    main()
