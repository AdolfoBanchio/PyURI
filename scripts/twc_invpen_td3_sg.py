"""
Single fixed-config TD3 + BPTT + Surrogate Gradients training run on
InvertedPendulum-v5. Mirrors twc_invpen_ppo_sg.py — edit MACRO_CONFIG.
"""
import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import os
import math
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from fiuri import PyUriTwc_V2, CalibratedActor, TWC_JSON
from td3_flat import TD3Config, TD3Engine, td3_train
from mlp import TwinCritic
from twc import twc_io as io
from utils import SequenceBuffer


# ──────────────────────────────────────────────────────────────────────────────
# MACRO CONFIG — edit these values to test a single hyperparameter set.
# Starting point: PPO trial 54's winning region, mapped to TD3.
# ──────────────────────────────────────────────────────────────────────────────

MACRO_CONFIG = {
    # Training loop
    "max_train_steps":   700_000,
    "warmup_steps":      8_000,
    "batch_size":        32,
    "num_update_loops":  1,
    "update_every":      1,
    "seed":              54001,

    # Evaluation
    "eval_interval_episodes": 10,
    "eval_episodes":          50,

    # BPTT
    "sequence_length":   8,
    "burn_in_length":    2,

    # Optimisation
    "actor_lr":          7e-4,
    "critic_lr":         1e-3,
    "gamma":             0.99,
    "tau":               0.005,
    "policy_delay":      2,
    "target_noise":      0.2,
    "noise_clip":        0.5,

    # Exploration: Gaussian (uncorrelated). OU is kept for MCC.
    "noise_type":                 "gaussian",
    "gaussian_sigma_init":        0.5,
    "gaussian_sigma_end":         0.15,
    "gaussian_sigma_decay_steps": 250_000,

    # Output-head calibration (CalibratedActor)
    "calib_target_std":  2.36,
    "calib_max_scale":   5.0,

    # Per-group gradient clipping
    "max_grad_norm_weights":    1.0,
    "max_grad_norm_thresholds": 0.5,
    "max_grad_norm_decay":      0.5,

    # TWC + SG (lifted from PPO trial 54)
    "internal_steps":    3,
    "steepness_fire":    9.95,
    "steepness_gj":      10.48,
    "steepness_input":   3.48,
    "input_thresh":      0.0084,

    # Critic
    "critic_hidden_layers": 256,
    "replay_buffer_size":   100_000,

    # Misc
    "model_prefix":      "twc_td3_sg_actor_invpen",
    "log_dir_name":      "twc_invpen_td3_sg_fixed",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_env(seed: int, env_id: str = "InvertedPendulum-v5"):
    import gymnasium as gym
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def phi_invpen_gentle(obs) -> float:
    # Soft Gaussian potential on pole angle only. Max 0.1 at upright.
    angle = obs[1]
    return 0.1 * math.exp(-(angle / 0.15) ** 2)


def invpen_score_fn(m_ret, s_rate, s_suc, m_act, all_scores, all_steps):
    """Mean return over the last 6 evals, with a soft variance penalty."""
    window = np.array(all_scores[-6:], dtype=np.float64)
    mean   = float(window.mean())
    std    = float(window.std()) if len(window) > 1 else 0.0
    return mean - 0.3 * std


def build_td3_config(macro: dict) -> TD3Config:
    cfg = TD3Config()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    for k, v in macro.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    macro = MACRO_CONFIG
    cfg = build_td3_config(macro)

    seed = cfg.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_env(seed)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device     = torch.device(cfg.device)

    # Actor: V2 TWC wrapped with the learnable affine head.
    base = PyUriTwc_V2(
        config_json     = TWC_JSON,
        obs_encoder     = io.ipen_obs_to_potentials_v2,
        action_decoder  = io.twc_out_2_invpen_mean,
        internal_steps  = macro["internal_steps"],
        steepness_fire  = macro["steepness_fire"],
        steepness_gj    = macro["steepness_gj"],
        steepness_input = macro["steepness_input"],
        input_thresh    = macro["input_thresh"],
    )
    actor = CalibratedActor(base, action_dim=action_dim).to(device)

    critic = TwinCritic(state_dim=state_dim, action_dim=action_dim).to(device)

    # Calibrate before engine construction so the target net (deepcopy)
    # captures the calibrated head — saves the slow Polyak catch-up.
    calib = actor.calibrate(env, target_std=cfg.calib_target_std,
                            max_scale=cfg.calib_max_scale)
    print("Calibration:", {k: round(v, 4) for k, v in calib.items()})

    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    engine = TD3Engine(
        gamma                    = cfg.gamma,
        tau                      = cfg.tau,
        observation_space        = env.observation_space,
        action_space             = env.action_space,
        actor                    = actor,
        critic                   = critic,
        actor_optimizer          = actor_opt,
        critic_optimizer         = critic_opt,
        policy_delay             = cfg.policy_delay,
        target_policy_noise      = cfg.target_noise,
        target_noise_clip        = cfg.noise_clip,
        max_grad_norm_weights    = cfg.max_grad_norm_weights,
        max_grad_norm_thresholds = cfg.max_grad_norm_thresholds,
        max_grad_norm_decay      = cfg.max_grad_norm_decay,
        device                   = device,
    )

    replay_buf = SequenceBuffer(capacity=cfg.replay_buffer_size, device=device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir   = f"out/runs/{macro['log_dir_name']}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    writer    = SummaryWriter(log_dir)

    for k, v in calib.items():
        writer.add_scalar(k, v, 0)

    with open(os.path.join(log_dir, "full_config.json"), "w") as f:
        f.write(cfg.to_json())

    print(f"Logging to {log_dir}")
    print(cfg.to_json())

    try:
        td3_train(
            env            = env,
            replay_buf     = replay_buf,
            engine         = engine,
            writer         = writer,
            timestamp      = timestamp,
            config         = cfg,
            phi            = phi_invpen_gentle,
            model_score_fn = invpen_score_fn,
            log_interval   = 200,
            invert_eval    = True,   # InvPen success = episode survives to truncation
        )
    finally:
        env.close()
        writer.close()


if __name__ == "__main__":
    main()
