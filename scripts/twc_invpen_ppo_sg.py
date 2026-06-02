"""
Single fixed-config PPO+SG training run on InvertedPendulum-v5.

Mirrors the structure of twc_mcc_td3.py but for PPO and the V2 TWC actor.
Useful for sanity-checking the training pipeline end-to-end before
launching the Optuna study in twc_optuna_invpen_ppo_sg.py.

The hyperparameter set below corresponds to the "median Optuna params"
suggested in the analysis (moderate steepness, mid-range learning rates,
internal_steps=4). Edit MACRO_CONFIG to try other combinations.
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

from fiuri import PyUriTwc_V2, TWC_JSON
from ppo.ppo import PPOConfig, PPOEngine, ppo_train
from mlp import ValueCriticInvPen
from twc import twc_io as io


# ──────────────────────────────────────────────────────────────────────────────
# MACRO CONFIG — edit these values to test a single hyperparameter set
# ──────────────────────────────────────────────────────────────────────────────

MACRO_CONFIG = {
    "max_train_steps": 700000,
    "episodes_per_update": 9,
    "ppo_epochs": 9,
    "mini_batch_size": 256,
    "seed": 5001,
    "eval_interval_updates": 5,
    "eval_episodes": 50,
    "burn_in_length": 2,
    "actor_lr": 0.0009341344007898773,
    "critic_lr": 0.0011499354899328403,
    "gamma": 0.99,
    "lam": 0.95,
    "clip_eps": 0.2,
    "entropy_coef": 0.005,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "log_std_init": 0.025430882155720314,
    "log_std_min": -3.0,
    "log_std_max": 0.5,
    "calib_target_std": 1.8217366146420804,
    "calib_max_scale": 5.0,

    # TWC-specific
    "internal_steps":          3,
    "steepness_fire": 8.674414895855056,
    "steepness_gj": 9.375720624646757,
    "steepness_input": 4.312114710091571,
    "input_thresh": 0.04341310710688284,

    # Misc
    "model_prefix":           "twc_ppo_SG_actor_invpen",
    "log_dir_name":           "twc_ppo_SG_invpen",
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
    # Same shaping as scripts/twc_optuna_invpen_ppo_sg.py — required to
    # reproduce trial 54 exactly.
    angle = obs[1]
    return 0.1 * math.exp(-(angle / 0.15) ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def build_ppo_config(macro: dict) -> PPOConfig:
    cfg = PPOConfig()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    for k, v in macro.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main():
    macro = MACRO_CONFIG
    cfg = build_ppo_config(macro)

    seed = cfg.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_env(seed)

    state_dim = env.observation_space.shape[0]
    device    = torch.device(cfg.device)

    actor = PyUriTwc_V2(
        config_json     = TWC_JSON,
        obs_encoder     = io.ipen_obs_to_potentials_v2,
        action_decoder  = io.twc_out_2_invpen_mean,
        internal_steps  = macro["internal_steps"],
        steepness_fire  = macro["steepness_fire"],
        steepness_gj    = macro["steepness_gj"],
        steepness_input = macro["steepness_input"],
        input_thresh    = macro["input_thresh"],
    )
    critic     = ValueCriticInvPen(state_dim=state_dim)
    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    actor.to(device)
    critic.to(device)

    engine = PPOEngine(
        actor            = actor,
        critic           = critic,
        action_space     = env.action_space,
        actor_optimizer  = actor_opt,
        critic_optimizer = critic_opt,
        config           = cfg,
        device           = device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir   = f"out/runs/{macro['log_dir_name']}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    writer    = SummaryWriter(log_dir)

    with open(os.path.join(log_dir, "full_config.json"), "w") as f:
        f.write(cfg.to_json())

    print(f"Logging to {log_dir}")
    print(cfg.to_json())

    try:
        all_eval_returns, all_eval_steps, best_model_path = ppo_train(
            env=env,
            engine=engine,
            writer=writer,
            timestamp=timestamp,
            config=cfg,
            trial=None,
            phi=phi_invpen_gentle,
        )
        print(f"\nFinal eval returns (last 6): {all_eval_returns[-6:]}")
        print(f"Best model: {best_model_path}")
    finally:
        env.close()
        writer.close()


if __name__ == "__main__":
    main()
