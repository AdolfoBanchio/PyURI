import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import os
import json
import gymnasium as gym
import numpy as np
import torch
import optuna
import ast
import torch.nn.functional as F
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, asdict
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from tqdm import tqdm
from utils import OUNoise, SequenceBuffer
from mlp import TwinCritic
from fiuri import PyUriTwc, build_fiuri_twc
from td3_flat import TD3Config, TD3Engine, td3_train

def make_env(seed, env_id="MountainCarContinuous-v0"):
    import gymnasium as gym
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

def objective(trial: optuna.Trial, study_name: str):
    cfg = TD3Config()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Shorter runs for search; keep evaluation cadence stable
    cfg.max_train_steps = 250_000
    cfg.warmup_steps = 10_000
    cfg.eval_interval_episodes = 10
    cfg.eval_episodes = 10
    cfg.num_update_loops = 1
    cfg.update_every = 1
    cfg.batch_size = 256
    cfg.model_prefix = "td3_flat_actor_noSG"
    cfg.use_SG = False
    # --- Tunable Hyperparameters ---
    cfg.actor_lr = trial.suggest_float("actor_lr", 1.5e-4, 4.0e-4, log=True)
    cfg.critic_lr = trial.suggest_float("critic_lr", 1.5e-4, 4.0e-4, log=True)
    cfg.gamma = trial.suggest_float("gamma", 0.978, 0.993)
    cfg.tau = trial.suggest_float("tau", 5e-3, 1.2e-2)
    cfg.target_noise = trial.suggest_float("target_noise", 0.20, 0.36)
    cfg.noise_clip = trial.suggest_float("noise_clip", 0.25, 0.45)
    cfg.ou_sigma_init = trial.suggest_float("sigma_start", 0.30, 0.50)
    cfg.ou_sigma_end = trial.suggest_float("sigma_end", 0.05, 0.12)

    # Per-trial seed to reduce correlation across samples
    seed = cfg.seed + trial.number
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_env(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = build_fiuri_twc()
    critic = TwinCritic(state_dim=state_dim, action_dim=action_dim)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    engine = TD3Engine(
        gamma=cfg.gamma,
        tau=cfg.tau,
        observation_space=env.observation_space,
        action_space=env.action_space,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        policy_delay=cfg.policy_delay,
        target_policy_noise=cfg.target_noise,
        target_noise_clip=cfg.noise_clip,
        device=cfg.device,
    )

    replay_buf = SequenceBuffer(capacity=cfg.replay_buffer_size)

    cfg.ou_sigma_decay_steps = int(cfg.max_train_steps * 0.7)
    noise = OUNoise(
        size=env.action_space.shape,
        mu=0.0,
        theta=0.15,
        sigma_init=cfg.ou_sigma_init,
        sigma_min=cfg.ou_sigma_end,
        decay_steps=cfg.ou_sigma_decay_steps,
        dt=1.0,
        seed=seed,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"trial_{trial.number}_{timestamp}"
    log_dir = f"out/runs/optuna/{study_name}/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    with open(os.path.join(log_dir, "trial_params.json"), "w") as f:
        json.dump(trial.params, f, indent=4)

    with open(os.path.join(log_dir, "full_config.json"), "w") as f:
        f.write(cfg.to_json())

    try:
        all_eval_scores, best_model_path = td3_train(
            env=env,
            replay_buf=replay_buf,
            engine=engine,
            writer=writer,
            timestamp=timestamp,
            config=cfg,
            trial=trial,
            OUNoise=noise,
        )

        final_stability_score = np.mean(all_eval_scores[-5:]) 
        return final_stability_score
    except optuna.TrialPruned:
        return -np.inf
    finally:
        env.close()
        writer.close()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = "twc_mcc_td3_flat_noSG_optuna"

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=1,
    )

    sampler = TPESampler(
        multivariate=True,
        group=True,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:///db.sqlite3",
        study_name=study_name,
        load_if_exists=True,
    )

    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if len(study.trials) == 0:
        baseline = {
            "actor_lr": 0.00024319130600441337,
            "critic_lr": 0.00028664900582161976,
            "gamma": 0.9922798944951167,
            "tau": 0.006443939133870398,
            "target_noise": 0.35757304576778454,
            "noise_clip": 0.3919642928588686,
            "sigma_start": 0.3244395391061171,
            "sigma_end": 0.11638525345225356,
        }
        study.enqueue_trial(baseline)
    elif len(completed_trials) > 0:
        best_params = study.best_trial.params
        print("\nEnqueuing previous best params for re-evaluation...")
        study.enqueue_trial(best_params)

    obj_wrapper = lambda trial: objective(trial, study_name)

    try:
        study.optimize(
            obj_wrapper,
            n_trials=20,
            gc_after_trial=True,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Optuna study interrupted by user.")

    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if completed_trials:
        print(f"\nStudy '{study_name}' complete.")
        print(f"Best value (max eval return): {study.best_value:.2f}")
        print("Best params:")
        print(json.dumps(study.best_params, indent=4))

        out_dir = f"out/runs/optuna/{study_name}_{timestamp}"
        os.makedirs(out_dir, exist_ok=True)
        best_params_path = os.path.join(out_dir, "best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(study.best_params, f, indent=4)
        print(f"Saved to {best_params_path}")
    else:
        print(f"\nStudy '{study_name}' has no completed trials yet (still running in other workers?).")


if __name__ == "__main__":
    main()
