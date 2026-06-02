"""
Standalone diagnostic: sweep SG steepness combinations and measure
how strong the gradient signal w.r.t. PyUriTwc_V2 parameters is.

Goal: confirm the Optuna search ranges in twc_optuna_invpen_ppo_sg.py
    steepness_fire   ∈ [4, 25]
    steepness_gj     ∈ [4, 25]
    steepness_input  ∈ [2, 12]
do not produce collapsed (vanishing) gradients on the actor's mean output.

For each (steepness_fire, steepness_gj, steepness_input) combination, we:
    1. Build a fresh PyUriTwc_V2 actor.
    2. Sample a batch of random InvertedPendulum-v5 observations into a
       (B, T, obs_dim) tensor.
    3. Run forward_bptt to get mean actions of shape (B, T, 1).
    4. Define a synthetic scalar loss = means.pow(2).mean().
    5. Backprop and record grad-norms on the three learnable tensors:
       weights, thresholds, decay.
    6. Also record ||action mean|| so we can spot saturated readouts.

Interpretation cheatsheet:
    - ||grad weights|| < 1e-6 across the board  → collapsed; that combo unusable.
    - ||grad||/max(|grad|) close to 1           → gradient concentrated on
                                                  one edge (most others dead).
    - Monotonic drop as steepness rises         → upper end of Optuna range
                                                  is hostile to learning.
    - ||action mean|| ≈ 0 regardless of steep   → readout-saturation problem
                                                  dominates; SG won't help on
                                                  its own (see PPOEngine head).

Run:
    python scripts/diagnose_sg_gradients.py
    python scripts/diagnose_sg_gradients.py --internal-steps 5
"""
import sys
import argparse
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import itertools
import torch
import gymnasium as gym

from fiuri import PyUriTwc_V2, TWC_JSON
from twc import twc_io as io


def measure_grads(
    steep_fire:     float,
    steep_gj:       float,
    steep_input:    float,
    input_thresh:   float = 0.01,
    internal_steps: int   = 1,
    B:              int   = 8,
    T:              int   = 32,
    seed:           int   = 0,
    device:         str   = "cpu",
) -> dict:
    """
    Build a fresh V2 actor at the requested steepness, push a random
    obs sequence through forward_bptt, backprop a synthetic scalar
    loss, and report gradient norms per learnable tensor.
    """
    torch.manual_seed(seed)
    env = gym.make("InvertedPendulum-v5")
    env.reset(seed=seed)

    actor = PyUriTwc_V2(
        config_json     = TWC_JSON,
        obs_encoder     = io.ipen_obs_to_potentials,
        action_decoder  = io.twc_out_2_invpen_mean,
        internal_steps  = internal_steps,
        steepness_fire  = steep_fire,
        steepness_gj    = steep_gj,
        steepness_input = steep_input,
        input_thresh    = input_thresh,
    ).to(device)
    actor.train()

    # (B, T, obs_dim) batch of plausible observations
    obs_seq = torch.stack([
        torch.as_tensor(env.observation_space.sample(), dtype=torch.float32)
        for _ in range(B * T)
    ]).view(B, T, -1).to(device)

    actor.reset(batch_size=B)
    means, _ = actor.forward_bptt(obs_seq)            # (B, T, 1)

    # Synthetic scalar loss that exercises every output dim.
    # mean.pow(2).mean() gives a smooth signal on the action magnitude.
    loss = means.pow(2).mean()
    loss.backward()

    def gnorm(p):
        if p.grad is None:
            return (0.0, 0.0)
        return (p.grad.norm().item(), p.grad.abs().max().item())

    nw, mw = gnorm(actor.weights)
    nt, mt = gnorm(actor.thresholds)
    nd, md = gnorm(actor.decay)

    env.close()
    return {
        "grad_w_norm":   nw,
        "grad_w_max":    mw,
        "grad_thr_norm": nt,
        "grad_thr_max":  mt,
        "grad_dec_norm": nd,
        "grad_dec_max":  md,
        "mean_abs":      means.detach().abs().mean().item(),
        "mean_std":      means.detach().std().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-steps", type=int, default=1,
                        help="Inner physics-step iterations per env step.")
    parser.add_argument("--batch",   type=int, default=8)
    parser.add_argument("--seq",     type=int, default=32)
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--device",  type=str, default="cpu")
    args = parser.parse_args()

    # Grid aligned with Optuna ranges in twc_optuna_invpen_ppo_sg.py.
    fire_grid  = [4.0, 10.0, 18.0, 25.0]
    gj_grid    = [4.0, 10.0, 18.0, 25.0]
    input_grid = [2.0, 6.0, 12.0]

    print(f"Internal steps: {args.internal_steps}   "
          f"batch={args.batch}  seq_len={args.seq}  seed={args.seed}")
    header = (
        f"{'sf':>5} {'sgj':>5} {'sin':>4} | "
        f"{'||∇w||':>9} {'max|∇w|':>9} | "
        f"{'||∇thr||':>9} {'||∇dec||':>9} | "
        f"{'|mean|':>9} {'std(mean)':>10}"
    )
    print(header)
    print("-" * len(header))

    for sf, sg, si in itertools.product(fire_grid, gj_grid, input_grid):
        r = measure_grads(
            steep_fire     = sf,
            steep_gj       = sg,
            steep_input    = si,
            internal_steps = args.internal_steps,
            B              = args.batch,
            T              = args.seq,
            seed           = args.seed,
            device         = args.device,
        )
        print(
            f"{sf:>5.1f} {sg:>5.1f} {si:>4.1f} | "
            f"{r['grad_w_norm']:>9.2e} {r['grad_w_max']:>9.2e} | "
            f"{r['grad_thr_norm']:>9.2e} {r['grad_dec_norm']:>9.2e} | "
            f"{r['mean_abs']:>9.2e} {r['mean_std']:>10.2e}"
        )


if __name__ == "__main__":
    main()
