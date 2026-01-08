import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fiuri import build_fiuri_twc, build_fiuri_twc_v2, PyUriTwc  # noqa: E402
from fiuri.gpu_opt import TWC_JSON  # noqa: E402
from td3_flat import TD3Config  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare PyUriTwc parameters between two saved models in a run directory."
        )
    )
    parser.add_argument(
        "models_dir",
        type=str,
        help="Directory containing a TD3 config (.json) and one or more .pth files.",
    )
    parser.add_argument(
        "--use-sg",
        action="store_true",
        help="Load the surrogate-gradient version of PyUriTwc if the run used it.",
    )
    parser.add_argument(
        "--model-a",
        type=str,
        default=None,
        help="Name (stem) of the first .pth file to compare. Defaults to 'best' if present.",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default=None,
        help="Name (stem) of the second .pth file to compare. Defaults to 'final' if present.",
    )
    return parser.parse_args()


def load_config_from_dir(models_dir: Path) -> TD3Config:
    """Load TD3 configuration from a single .json file in the directory."""
    json_files = sorted(models_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json configuration file found in {models_dir}")

    if len(json_files) == 1:
        cfg_file = json_files[0]
    else:
        cfg_candidates = [f for f in json_files if "config" in f.stem.lower()]
        if not cfg_candidates:
            raise ValueError(f"Config .json file not found in {models_dir}")
        if len(cfg_candidates) > 1:
            raise ValueError(
                f"Multiple config .json files found in {models_dir}: {[f.name for f in cfg_candidates]}"
            )
        cfg_file = cfg_candidates[0]

    cfg = TD3Config()
    with cfg_file.open("r") as f:
        cfg = cfg.load(json.load(f))
    return cfg


def build_actor(cfg: TD3Config, use_sg: bool) -> PyUriTwc:
    if use_sg:
        return build_fiuri_twc_v2(
            steepness_fire=cfg.steepness_fire,
            steepness_gj=cfg.steepness_gj,
            steepness_input=cfg.steepness_input,
            input_thresh=cfg.input_thresh,
        )
    return build_fiuri_twc()


def load_robust_model(model: PyUriTwc, model_path: Path) -> PyUriTwc:
    """Load a state dict even if it was saved from a torch.compile() model."""
    state_dict = torch.load(model_path, map_location=DEVICE)
    clean_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "") if key.startswith("_orig_mod.") else key
        clean_state_dict[new_key] = value

    model.load_state_dict(clean_state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def load_all_models(models_dir: Path, cfg: TD3Config, use_sg: bool) -> Dict[str, PyUriTwc]:
    models = {}
    for model_path in sorted(models_dir.glob("*.pth")):
        actor = build_actor(cfg, use_sg)
        models[model_path.stem] = load_robust_model(actor, model_path)
    if not models:
        raise FileNotFoundError(f"No .pth models found in {models_dir}")
    return models


def select_models(model_names, preferred_a=None, preferred_b=None) -> Tuple[str, str]:
    names = sorted(model_names)

    def find_by_hint(hint: str):
        for name in names:
            if hint in name.lower():
                return name
        return None

    if preferred_a and preferred_b:
        if preferred_a not in model_names or preferred_b not in model_names:
            raise ValueError(
                f"Requested models '{preferred_a}' and '{preferred_b}' not found. "
                f"Available: {', '.join(names)}"
            )
        return preferred_a, preferred_b

    best = find_by_hint("first")
    final = find_by_hint("final")
    if best and final and best != final:
        return best, final

    if len(names) < 2:
        raise ValueError("Need at least two models to compare.")
    return names[0], names[1]


def compare_neuron_params(model_a: PyUriTwc, model_b: PyUriTwc):
    idx_to_name = {idx: name for name, idx in model_a.neuron_names.items()}
    thresholds_a = model_a.thresholds.detach().cpu()
    thresholds_b = model_b.thresholds.detach().cpu()
    decay_a = model_a.decay.detach().cpu()
    decay_b = model_b.decay.detach().cpu()

    print("\nNeuron thresholds and decay:")
    for idx in range(len(idx_to_name)):
        name = idx_to_name[idx]
        th_a, th_b = thresholds_a[idx].item(), thresholds_b[idx].item()
        dc_a, dc_b = decay_a[idx].item(), decay_b[idx].item()
        print(
            f"- {name:4s} | threshold: {th_a: .6f} -> {th_b: .6f} (delta {th_b - th_a:+.6f}) "
            f"| decay: {dc_a: .6f} -> {dc_b: .6f} (delta {dc_b - dc_a:+.6f})"
        )


def compare_connections(model_a: PyUriTwc, model_b: PyUriTwc, config_json: dict):
    weights_a = model_a.weights.detach().cpu()
    weights_b = model_b.weights.detach().cpu()
    name_to_idx = model_a.neuron_names

    print("\nConnections (weight src -> dst):")
    for edge in config_json["edges"]:
        src, dst, ctype = edge["src"], edge["dst"], edge["type"]
        src_idx, dst_idx = name_to_idx[src], name_to_idx[dst]
        w_a = weights_a[dst_idx, src_idx].item()
        w_b = weights_b[dst_idx, src_idx].item()
        print(
            f"- {src:3s} -> {dst:3s} [{ctype}] : {w_a: .6f} -> {w_b: .6f} (delta {w_b - w_a:+.6f})"
        )


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    if not models_dir.is_dir():
        raise NotADirectoryError(f"{models_dir} is not a directory.")

    cfg = load_config_from_dir(models_dir)
    models = load_all_models(models_dir, cfg, args.use_sg)

    chosen_a, chosen_b = select_models(
        models.keys(), preferred_a=args.model_a, preferred_b=args.model_b
    )
    model_a, model_b = models[chosen_a], models[chosen_b]

    print(f"Comparing models '{chosen_a}' and '{chosen_b}' in {models_dir}")
    compare_neuron_params(model_a, model_b)
    compare_connections(model_a, model_b, TWC_JSON)


if __name__ == "__main__":
    main()
