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
from datetime import datetime
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from torch.utils.tensorboard import SummaryWriter
from utils import SequenceBuffer
from mlp import TwinCriticInvPen
from fiuri import build_fiuri_twc_invpen, PyUriTwc, TWC_JSON
from td3_flat import TD3Engine, TD3Config, td3_train
from twc import twc_io as io

def make_env(seed, env_id="InvertedPendulum-v5"):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def phi_invpen(obs):
    angle = obs[1]
    ang_vel = obs[3]
    angle_term = 1.0 - min(abs(angle) / 0.2, 1.0)
    vel_term = 1.0 - min(abs(ang_vel) / 2.0, 1.0)  # tune the 2.0 bound
    return float(angle_term * vel_term)


def objective(trial: optuna.Trial, study_name: str):
    cfg = TD3Config()
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.max_train_steps = 300_000
    cfg.warmup_steps = 5000
    cfg.eval_interval_episodes = 20
    cfg.eval_episodes = 50
    cfg.update_every = 1
    cfg.batch_size = 256
    cfg.model_prefix = "td3_actor_invpen"
    cfg.policy_delay = 2

    # --- Tunable Hyperparameters ---
    cfg.num_update_loops = trial.suggest_int("num_update_loops", 1, 3)
    cfg.sequence_length = trial.suggest_categorical("sequence_length", [16, 20, 24])
    cfg.burn_in_length = 8

    cfg.actor_lr = trial.suggest_float("actor_lr", 1.5e-4, 5e-4, log=True)
    cfg.critic_lr = trial.suggest_float("critic_lr", 1.5e-4, 5e-4, log=True)

    cfg.gamma = trial.suggest_float("gamma", 0.96, 0.992)
    cfg.tau = trial.suggest_float("tau", 0.005, 0.01)

    cfg.target_noise = trial.suggest_float("target_noise", 0.20, 0.35)
    cfg.noise_clip = trial.suggest_float("noise_clip", cfg.target_noise + 0.05, 0.50)

    cfg.ou_sigma_init = trial.suggest_float("sigma_start", 0.35, 0.60)
    cfg.ou_sigma_end = trial.suggest_float("sigma_end", 0.01, 0.10)

    cfg.ou_sigma_decay_steps = int(cfg.max_train_steps * 0.60)

    seed = trial.suggest_int("seed", 0, 2_000_000_000)
    cfg.seed = seed

    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_env(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = PyUriTwc(config_json=TWC_JSON,
                     obs_encoder=io.ipen_obs_to_potentials_v2,
                     action_decoder=io.twc_out_2_invpen_action)
    
    critic = TwinCriticInvPen(state_dim=state_dim, action_dim=action_dim)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    actor.to(cfg.device)
    critic.to(cfg.device)
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

    replay_buf = SequenceBuffer(capacity=cfg.replay_buffer_size, device=cfg.device)

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
        # Default model_score_fn (mean return) — do not pass model_score_fn.
        all_eval_scores, all_eval_steps, best_model_path = td3_train(
            env=env,
            replay_buf=replay_buf,
            engine=engine,
            writer=writer,
            timestamp=timestamp,
            config=cfg,
            trial=trial,
            phi=phi_invpen,
            log_interval=200,
            invert_eval=True, # for inverted pendulum if episode truncated == success, else it fell. 
        )

        # Stability window: mean of last 6 eval returns.
        final_rew = float(np.mean(all_eval_scores[-6:]))

        # Failure zone: agent never learned to balance for any meaningful duration.
        if final_rew < 100:
            return -100.0

        return final_rew

    except optuna.TrialPruned:
        return -200.0
    finally:
        env.close()
        writer.close()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = "twc_invpen_td3_flat_noSG"

    sampler = TPESampler(
        multivariate=True,
        group=True,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        storage="sqlite:///td3_invpen.sqlite3",
        study_name=study_name,
        load_if_exists=True,
    )

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
