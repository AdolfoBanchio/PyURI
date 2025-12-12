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

def debug_encoder_comparison(num_samples: int = 1000, seed: int = 0):
    """Compare encoder outputs directly over multiple samples and report mean error per interface."""
    xml_path = Path(__file__).parent / 'TWFiuriBaseFIU.xml'
    fiu_twc = FiuModel('FIU')
    fiu_twc.loadFromFile(str(xml_path))

    rng = np.random.default_rng(seed)
    observations = rng.uniform(-1.0, 1.0, size=(num_samples, 2)).astype(np.float32)

    errors = {"PLM": [], "AVM": [], "ALM": [], "PVD": []}

    for obs in observations:
        fiu_twc.Reset()
        fiu_twc.interfaces['IN1'].setValue(float(obs[0]))
        fiu_twc.interfaces['IN2'].setValue(float(obs[1]))
        fiu_twc.interfaces['IN1'].feedNN()
        fiu_twc.interfaces['IN2'].feedNN()

        ariel_vals = {
            "PLM": fiu_twc.getNeuron('PLM').getInternalState(),
            "AVM": fiu_twc.getNeuron('AVM').getInternalState(),
            "ALM": fiu_twc.getNeuron('ALM').getInternalState(),
            "PVD": fiu_twc.getNeuron('PVD').getInternalState(),
        }

        obs_tensor = torch.tensor([obs], dtype=torch.float32)
        ex_in, in_in = mcc_obs_encoder(obs_tensor)

        # Input order: [PVD, PLM, AVM, ALM]
        torch_vals = {
            "PVD": (ex_in[0, 0] + in_in[0, 0]).item(),
            "PLM": (ex_in[0, 1] + in_in[0, 1]).item(),
            "AVM": (ex_in[0, 2] + in_in[0, 2]).item(),
            "ALM": (ex_in[0, 3] + in_in[0, 3]).item(),
        }

        for name in errors:
            errors[name].append(abs(ariel_vals[name] - torch_vals[name]))

    print(f"Mean absolute encoder error over {num_samples} samples in [-1, 1]:")
    for name, vals in errors.items():
        mean_err = float(np.mean(vals)) if vals else 0.0
        print(f"  {name}: {mean_err}")

def debug_decoder_comparison(num_samples: int = 1000, seed: int = 0):
    """Compare decoder outputs using Ariel REV/FWD internal states over many samples."""
    xml_path = Path(__file__).parent / 'TWFiuriBaseFIU.xml'
    fiu_twc = FiuModel('FIU')
    fiu_twc.loadFromFile(str(xml_path))

    rng = np.random.default_rng(seed)
    test_obs = rng.uniform(-1.0, 1.0, size=(num_samples, 2)).astype(np.float32)

    diffs = []

    print("\nDecoder comparison (Ariel vs Torch mapping):")
    for obs in test_obs:
        fiu_twc.Reset()
        ariel_out = fiu_twc.Update(obs, mode=None, doLog=False)

        rev_int = fiu_twc.getNeuron('REV').getInternalState()
        fwd_int = fiu_twc.getNeuron('FWD').getInternalState()

        y = torch.tensor([[rev_int, fwd_int]], dtype=torch.float32)
        torch_out = twc_out_2_mcc_action(y).squeeze().item()

        diffs.append(abs(ariel_out - torch_out))

    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    print(f"Mean absolute decoder error over {num_samples} samples in [-1, 1]: {mean_diff}")

if __name__ == "__main__":
    debug_encoder_comparison()
    debug_decoder_comparison()
