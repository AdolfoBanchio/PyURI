"""
Optuna study: TD3 + BPTT + Surrogate Gradients on InvertedPendulum-v5.
Search ranges are narrowed around the PPO trial-54 winning region.
"""
import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import os
import json
import math
import numpy as np
import torch
import optuna
from datetime import datetime
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from torch.utils.tensorboard import SummaryWriter

from fiuri import PyUriTwc_V2, CalibratedActor, TWC_JSON
from td3_flat import TD3Config, TD3Engine, td3_train
from mlp import TwinCritic
from twc import twc_io as io
from utils import SequenceBuffer


def make_env(seed: int, env_id: str = "InvertedPendulum-v5"):
    import gymnasium as gym
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def phi_invpen_gentle(obs) -> float:
    angle = obs[1]
    return 0.1 * math.exp(-(angle / 0.15) ** 2)


def invpen_score_fn(m_ret, s_rate, s_suc, m_act, all_scores, all_steps):
    """Mean return over the last 6 evals; soft variance penalty."""
    window = np.array(all_scores[-6:], dtype=np.float64)
    mean   = float(window.mean())
    std    = float(window.std()) if len(window) > 1 else 0.0
    return mean - 0.3 * std


# Same dead-init floor used in the PPO study.
DEAD_INIT_RAW_STD_FLOOR = 0.03


def objective(trial: optuna.Trial, study_name: str) -> float:
    cfg = TD3Config()
    cfg.device              = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.max_train_steps     = 700_000
    cfg.warmup_steps        = 8_000
    cfg.batch_size          = 32
    cfg.num_update_loops    = 1
    cfg.update_every        = 1
    cfg.eval_interval_episodes = 10
    cfg.eval_episodes       = 50
    cfg.burn_in_length      = 2
    cfg.policy_delay        = 2
    cfg.target_noise        = 0.2
    cfg.noise_clip          = 0.5
    cfg.gamma               = 0.99
    cfg.noise_type          = "gaussian"
    cfg.replay_buffer_size  = 100_000
    cfg.critic_hidden_layers = 256
    cfg.model_prefix        = "twc_td3_sg_invpen"

    # Per-group gradient clipping (fixed at the productive defaults).
    cfg.max_grad_norm_weights    = 1.0
    cfg.max_grad_norm_thresholds = 0.5
    cfg.max_grad_norm_decay      = 0.5

    # ── Searched hyperparameters ──────────────────────────────────────────
    cfg.actor_lr  = trial.suggest_float("actor_lr",  5e-4, 1e-3, log=True)
    cfg.critic_lr = trial.suggest_float("critic_lr", 5e-4, 2e-3, log=True)
    cfg.tau       = trial.suggest_float("tau",       3e-3, 1e-2, log=True)

    cfg.gaussian_sigma_init        = trial.suggest_float("sigma_init", 0.4, 0.8)
    cfg.gaussian_sigma_end         = 0.15
    cfg.gaussian_sigma_decay_steps = trial.suggest_int("sigma_decay_steps", 150_000, 350_000)

    cfg.calib_target_std = trial.suggest_float("calib_target_std", 1.8, 2.6)
    cfg.calib_max_scale  = 5.0

    cfg.sequence_length = trial.suggest_categorical("sequence_length", [6, 8, 12])

    # TWC-specific. Narrowed around PPO trial 54.
    internal_steps   = 3
    steepness_fire   = trial.suggest_float("steepness_fire",  8.0, 14.0)
    steepness_gj     = trial.suggest_float("steepness_gj",    6.0, 12.0)
    steepness_input  = trial.suggest_float("steepness_input", 3.0,  6.0)
    input_thresh     = trial.suggest_float("input_thresh",    0.005, 0.05, log=True)

    # Deterministic per-trial seed.
    seed     = trial.number * 1000 + 1
    cfg.seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_env(seed)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device     = torch.device(cfg.device)

    base = PyUriTwc_V2(
        config_json     = TWC_JSON,
        obs_encoder     = io.ipen_obs_to_potentials_v2,
        action_decoder  = io.twc_out_2_invpen_mean,
        internal_steps  = internal_steps,
        steepness_fire  = steepness_fire,
        steepness_gj    = steepness_gj,
        steepness_input = steepness_input,
        input_thresh    = input_thresh,
    )
    actor  = CalibratedActor(base, action_dim=action_dim).to(device)
    critic = TwinCritic(state_dim=state_dim, action_dim=action_dim).to(device)

    # Logging — set up early so we can log calibration even if we bail.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"trial_{trial.number}_{timestamp}"
    log_dir   = f"out/runs/optuna/{study_name}/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    writer    = SummaryWriter(log_dir)

    with open(os.path.join(log_dir, "trial_params.json"), "w") as f:
        json.dump(trial.params, f, indent=4)
    with open(os.path.join(log_dir, "full_config.json"), "w") as f:
        f.write(cfg.to_json())

    # Calibrate before engine construction so deepcopy captures the
    # calibrated head; also gate dead-init trials immediately.
    calib   = actor.calibrate(env, target_std=cfg.calib_target_std,
                              max_scale=cfg.calib_max_scale)
    raw_std = calib["calib/raw_std"]
    trial.set_user_attr("calib_raw_std", raw_std)
    for k, v in calib.items():
        writer.add_scalar(k, v, 0)
    if raw_std < DEAD_INIT_RAW_STD_FLOOR:
        env.close(); writer.close()
        return -50.0

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

    try:
        all_eval_scores, all_eval_steps, best_model_path = td3_train(
            env            = env,
            replay_buf     = replay_buf,
            engine         = engine,
            writer         = writer,
            timestamp      = timestamp,
            config         = cfg,
            trial          = trial,
            phi            = phi_invpen_gentle,
            model_score_fn = invpen_score_fn,
            log_interval   = 200,
            invert_eval    = True,
        )
        if len(all_eval_scores) < 2:
            return -200.0
        if np.mean(all_eval_scores[-6:]) < 8.0:
            return -100.0
        return invpen_score_fn(None, None, None, None, all_eval_scores, all_eval_steps)

    except optuna.TrialPruned:
        return -200.0
    finally:
        env.close()
        writer.close()


def main():
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = "twc_invpen_td3_SG"

    sampler = TPESampler(multivariate=True, group=True, n_startup_trials=8)
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials = 5,
        n_warmup_steps   = 20,
        interval_steps   = 4,
    )

    study = optuna.create_study(
        direction      = "maximize",
        sampler        = sampler,
        pruner         = pruner,
        storage        = "sqlite:///twc_td3_invpen_SG.sqlite3",
        study_name     = study_name,
        load_if_exists = True,
    )

    obj_wrapper = lambda trial: objective(trial, study_name)

    try:
        study.optimize(obj_wrapper, n_trials=40, gc_after_trial=True, show_progress_bar=True)
    except KeyboardInterrupt:
        print("Optuna study interrupted by user.")

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if completed:
        print(f"\nStudy '{study_name}' complete.")
        print(f"Best score: {study.best_value:.2f}")
        print("Best params:")
        print(json.dumps(study.best_params, indent=4))

        out_dir = f"out/runs/optuna/{study_name}_{timestamp}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "best_params.json"), "w") as f:
            json.dump(study.best_params, f, indent=4)
    else:
        print(f"\nStudy '{study_name}' has no completed trials yet.")


if __name__ == "__main__":
    main()
