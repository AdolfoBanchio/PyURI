import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import torch
import numpy as np
from twc.twc_builder import build_twc
from twc.twc_io import mcc_obs_encoder, twc_out_2_mcc_action
from ariel.Model import Model as FiuModel

def debug_encoder_comparison():
    """Compare encoder outputs directly."""
    xml_path = Path(__file__).parent / 'TWFiuriBaseFIU.xml'
    fiu_twc = FiuModel('FIU')
    fiu_twc.loadFromFile(str(xml_path))
    
    # Test observation
    obs = np.array([-0.5, 0.0])
    
    # Get ariel encoder output
    fiu_twc.Reset()
    fiu_twc.interfaces['IN1'].setValue(obs[0])
    fiu_twc.interfaces['IN2'].setValue(obs[1])
    fiu_twc.interfaces['IN1'].feedNN()
    fiu_twc.interfaces['IN2'].feedNN()
    
    ariel_plm = fiu_twc.getNeuron('PLM').getInternalState()
    ariel_avm = fiu_twc.getNeuron('AVM').getInternalState()
    ariel_alm = fiu_twc.getNeuron('ALM').getInternalState()
    ariel_pvd = fiu_twc.getNeuron('PVD').getInternalState()
    
    print("Ariel encoder outputs:")
    print(f"  PLM: {ariel_plm:.6f}")
    print(f"  AVM: {ariel_avm:.6f}")
    print(f"  ALM: {ariel_alm:.6f}")
    print(f"  PVD: {ariel_pvd:.6f}")
    
    # Get torch encoder output
    obs_tensor = torch.tensor([obs], dtype=torch.float32)
    ex_in, in_in = mcc_obs_encoder(obs_tensor)
    
    # Input order: [PVD, PLM, AVM, ALM]
    torch_pvd = (ex_in[0, 0] + in_in[0, 0]).item()
    torch_plm = (ex_in[0, 1] + in_in[0, 1]).item()
    torch_avm = (ex_in[0, 2] + in_in[0, 2]).item()
    torch_alm = (ex_in[0, 3] + in_in[0, 3]).item()
    
    print("\nTorch encoder outputs:")
    print(f"  PLM: {torch_plm:.6f}")
    print(f"  AVM: {torch_avm:.6f}")
    print(f"  ALM: {torch_alm:.6f}")
    print(f"  PVD: {torch_pvd:.6f}")
    
    print("\nDifferences:")
    print(f"  PLM: {abs(ariel_plm - torch_plm):.6f}")
    print(f"  AVM: {abs(ariel_avm - torch_avm):.6f}")
    print(f"  ALM: {abs(ariel_alm - torch_alm):.6f}")
    print(f"  PVD: {abs(ariel_pvd - torch_pvd):.6f}")

def debug_weights():
    """Check if weights are being set correctly."""
    xml_path = Path(__file__).parent / 'TWFiuriBaseFIU.xml'
    fiu_twc = FiuModel('FIU')
    fiu_twc.loadFromFile(str(xml_path))
    
    twc = build_twc(
        obs_encoder=mcc_obs_encoder,
        action_decoder=lambda x: x,
        internal_steps=1,
        log_stats=False,
    )
    
    from twc_validation import sync_weights_from_ariel_to_torch, get_neuron_names_from_json
    neuron_names = get_neuron_names_from_json()
    sync_weights_from_ariel_to_torch(fiu_twc, twc, neuron_names)
    
    # Check a few weights
    print("\nWeight check:")
    ariel_weights = {}
    for conn in fiu_twc.neuralnetwork.getConnections():
        src = conn.getSource().getName()
        dst = conn.getTarget().getName()
        key = f"{src}->{dst}"
        ariel_weights[key] = conn.getTestWeight()
    
    # Check PLM->AVD (should be IN connection)
    if 'PLM->AVD' in ariel_weights:
        print(f"  Ariel PLM->AVD: {ariel_weights['PLM->AVD']:.6f}")
        # Check torch weight
        input_idx = {name: i for i, name in enumerate(neuron_names['input'])}
        hidden_idx = {name: i for i, name in enumerate(neuron_names['hidden'])}
        plm_idx = input_idx['PLM']
        avd_idx = hidden_idx['AVD']
        torch_w_raw = twc.in2hid_IN.w[plm_idx, avd_idx].item()
        torch_w_softplus = torch.nn.functional.softplus(torch.tensor(torch_w_raw)).item()
        print(f"  Torch PLM->AVD (raw): {torch_w_raw:.6f}")
        print(f"  Torch PLM->AVD (softplus): {torch_w_softplus:.6f}")
        print(f"  Mask value: {twc.in2hid_IN.w_mask[plm_idx, avd_idx].item()}")

def debug_decoder_comparison():
    """Compare decoder outputs using Ariel REV/FWD internal states."""
    xml_path = Path(__file__).parent / 'TWFiuriBaseFIU.xml'
    fiu_twc = FiuModel('FIU')
    fiu_twc.loadFromFile(str(xml_path))

    # A few diverse observations within env bounds
    test_obs = [
        np.array([-0.5, 0.0], dtype=np.float32),
        np.array([0.2, 0.05], dtype=np.float32),
        np.array([-1.0, -0.07], dtype=np.float32),
        np.array([0.6, 0.1], dtype=np.float32),
    ]

    print("\nDecoder comparison (Ariel vs Torch mapping):")
    for i, obs in enumerate(test_obs):
        fiu_twc.Reset()
        ariel_out = fiu_twc.Update(obs, mode=None, doLog=False)

        rev_int = fiu_twc.getNeuron('REV').getInternalState()
        fwd_int = fiu_twc.getNeuron('FWD').getInternalState()

        y = torch.tensor([[rev_int, fwd_int]], dtype=torch.float32)
        torch_out = twc_out_2_mcc_action(y).squeeze().item()

        print(f"  Case {i+1}: obs={obs.tolist()} | REV={rev_int:.6f}, FWD={fwd_int:.6f}")
        print(f"    Ariel: {ariel_out:.6f} | Torch: {torch_out:.6f} | diff={abs(ariel_out - torch_out):.6f}")

if __name__ == "__main__":
    debug_encoder_comparison()
    debug_decoder_comparison()
    debug_weights()
