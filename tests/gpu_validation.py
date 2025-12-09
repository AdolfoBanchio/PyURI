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
import torch.functional as F
import numpy as np
import gymnasium as gym
from twc.twc_builder import build_twc, TWC
from twc.twc_io import (
    mcc_obs_encoder,
    twc_out_2_mcc_action,
)
from ariel.Model import Model as FiuModel
from ariel import Connection as con
from fiuri import PyUriTwc, PyUriTwc_V2, build_fiuri_twc_v2, build_fiuri_twc
from fiuri.gpu_opt import TWC_JSON as config

def sync_parameters(ariel_mod: FiuModel, twc_v2: PyUriTwc):
    """
    This functions copies all the parameters from torch version to original version
    """
    name_to_idx = twc_v2.neuron_names
    print(name_to_idx)
    with torch.no_grad():
        # Reset weights
        sp = torch.nn.functional.softplus

        # Neuron params
        for n in ariel_mod.neuralnetwork.getNeuronNames():
            idx = name_to_idx.get(n, None)
            if idx is None:
                continue
            neuron = ariel_mod.getNeuron(n)
            neuron.setTestThreshold(float(twc_v2.thresholds[idx]))
            neuron.setTestDecayFactor(float(twc_v2.decay[idx]))

        # Connection weights
        for conn in ariel_mod.neuralnetwork.getConnections():
            src = conn.getSource().getName()
            dst = conn.getTarget().getName()
            si = name_to_idx.get(src, None)
            di = name_to_idx.get(dst, None)
            if si is None or di is None:
                continue
            conn.setTestWeight(float(sp(twc_v2.weights[di, si])))

def run_test():
    print("--- Starting Comparison Test ---")
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # Load ariel model
    xml_path = os.path.join(Path(__file__).parent, 'TWFiuriBaseFIU.xml')
    fiu = FiuModel('FIU')
    fiu.loadFromFile(xml_path)
    fiu.Reset()

    # 2. Init New
    opt = build_fiuri_twc()
    opt.reset_state(1)
    
    opt2 = build_fiuri_twc_v2(steepness_fire=1,
                                  steepness_gj=1,
                                  steepness_input=1,
                                  input_thresh=0,
                                  leaky_slope=0.2)
    opt2.reset_state(1)
    
    opt2.load_state_dict(opt.state_dict(), strict=False)

    opt.to(device=device)
    opt2.to(device=device)
    opt.device = device
    opt2.device = device
    # 3. Sync
    sync_parameters(fiu, opt)
    
    # 4. Simulation
    steps = 1000
    trace = []
    hist_leg, hist_new, hist_new2 = [], [], []
    out_dir = os.path.join('out/tests/gpu_validation')

    # Create output file
    with open(os.path.join(out_dir,"comparison_trace.txt"), "w") as f:
        f.write("STEP | OBS (Pos, Vel) | Action_Leg | Action_New | Diff | Neurons (E_leg, E_new)...\n")
        
        for i in range(steps):

            if i % 5 == 0:
                fiu.Reset()
                opt.reset_internal_only()
                opt2.reset_internal_only()
            # Random Observation
            pos = np.random.uniform(-1.2, 0.6)
            vel = np.random.uniform(-0.07, 0.07)
            
            # --- Legacy Step ---
            act_leg = fiu.Update((pos, vel), mode=None, doLog=False)

            # --- New Step ---
            obs_t = torch.tensor([[pos, vel]], dtype=torch.float32)
            act_new = opt(obs_t).item()
            act_new2 = opt2(obs_t).item()
            
            # --- Log ---
            hist_leg.append(act_leg)
            hist_new.append(act_new)
            hist_new2.append(act_new2)
            
            diff = abs(act_leg - act_new)
            log_line = f"{i:3} | ({pos:5.2f}, {vel:5.2f}) | {act_leg:6.3f} | {act_new:6.3f} | {diff:6.3e} | "
            
            # Log specific neurons to debug
            neuron_debug = []
            for name in ["PLM", "AVA", "FWD"]: # Sample check
                idx = config['neurons'][name]
                e_leg = fiu.getNeuron(name).internalstate
                e_new = opt.stored_E[0, idx].item()
                neuron_debug.append(f"{name}: {e_leg:.1f}/{e_new:.1f}")
            
            f.write(log_line + " | ".join(neuron_debug) + "\n")

    print(f"Trace saved to comparison_trace.txt")
    
    # 5. Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist_leg, label='Legacy', alpha=0.7)
    plt.plot(hist_new, label='New', alpha=0.7, linestyle='--')
    plt.title("Action Trace")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(hist_leg, alpha=0.5, label='Legacy', bins=20)
    plt.hist(hist_new, alpha=0.5, label='Flat Version', bins=20, histtype='step', linewidth=2)
    plt.hist(hist_new2, alpha=0.5, label='Flat Version (SG)', bins=20, histtype='step', linewidth=2)
    plt.title("Action Histogram")
    plt.legend()
    
    plt.savefig(os.path.join(out_dir,'output_histogram.png'))
    print("Plot saved to output_histogram.png")

if __name__ == "__main__":
    run_test()