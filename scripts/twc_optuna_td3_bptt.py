import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import os
import itertools
import json
import gymnasium as gym
import numpy as np
import torch
import optuna
import optunahub
import optuna_distributed
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from functools import partial
from td3 import TD3Engine, TD3Config, td3_train
from utils import ReplayBuffer, OUNoise, SequenceBuffer
from mlp import Critic
from twc import (
    build_twc,
    mcc_obs_encoder,
    twc_out_2_mcc_action,
)

def make_env(seed, env_id="MountainCarContinuous-v0"):
    import gymnasium as gym
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

def objective(trial: optuna.Trial, study_name):
    cfg = TD3Config()
    # --- Set Fixed Parameters ---
    cfg.use_bptt = True 
    cfg.max_episode = 300
    cfg.max_time_steps = 999
    cfg.warmup_steps = 10_000
    cfg.eval_interval_episodes = 10
    cfg.eval_episodes = 10
    cfg.policy_delay = 2
    cfg.batch_size = 256
    
    # Models Parameters
    cfg.critic_hidden_layers = [400, 300]
    cfg.twc_internal_steps = 1
    cfg.rnd_init = True
    cfg.twc_trhesholds = [-0.5, 0.0, 0.0]
    cfg.twc_decays = [0.1, 0.1, 0.1]
    cfg.use_v2 = False
    
    # --- Set Tunable Hyperparameters ---
    cfg.sequence_length = trial.suggest_int("sequence_length", 8, 16)
    max_burn_in = cfg.sequence_length - 2
    cfg.burn_in_length = trial.suggest_int("burn_in_length", 4, max_burn_in)
    cfg.num_update_loops = trial.suggest_int("num_update_loops", 1, 3)
    
    cfg.actor_lr = trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True)
    cfg.critic_lr = trial.suggest_float("critic_lr", 1e-4, 3e-3, log=True)
    cfg.gamma = trial.suggest_float("gamma", 0.98, 0.999)
    cfg.tau = trial.suggest_float("tau", 0.001, 0.02, log=True)
    
    cfg.target_noise = trial.suggest_float("target_noise", 0.1, 0.3)
    cfg.noise_clip = trial.suggest_float("noise_clip", 0.3, 0.5)
    # (OU)
    cfg.sigma_start = trial.suggest_float("sigma_start", 0.2, 0.5)
    cfg.sigma_end = trial.suggest_float("sigma_end", 0.01, 0.1)
    cfg.sigma_decay_episodes = trial.suggest_int("sigma_decay_episodes", 150, 250)

    # V2 twc hyperparameters
    if cfg.use_v2:
        v2_params = {
            'steepness_fire': trial.suggest_float("steep_fire", 5.0, 25.0, log=True),
            'steepness_gj': trial.suggest_float("steep_gj", 5.0, 25.0, log=True),
            'steepness_input': trial.suggest_float("steep_input", 3.0, 20.0, log=True),
            'input_thresh': trial.suggest_float("input_thresh", 1e-3, 5e-2, log=True),
            'leaky_slope': trial.suggest_float("leaky_slope", 0.01, 0.2, log=True)
            }
    # Seed per trial
    seed = 42 + trial.number
    np.random.seed(seed); torch.manual_seed(seed)
    env = make_env(seed)

    # Build models per trial to avoid cross-trial state leakage
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = build_twc(
        obs_encoder=mcc_obs_encoder,
        action_decoder=twc_out_2_mcc_action,
        internal_steps=cfg.twc_internal_steps,
        initial_thresholds=cfg.twc_trhesholds,
        initial_decays=cfg.twc_decays,
        rnd_init=cfg.rnd_init,
        use_V2=cfg.use_v2,
        log_stats=False,
        **({'v2_params': v2_params} if cfg.use_v2 else {})
    )
    critic_1 = Critic(state_dim, action_dim, size=cfg.critic_hidden_layers)
    critic_2 = Critic(state_dim, action_dim, size=cfg.critic_hidden_layers)

    # Optimizers
    actor_opt = torch.optim.Adam(actor.parameters(),  lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(
        itertools.chain(critic_1.parameters(), critic_2.parameters()),
        lr=cfg.critic_lr,
    )

    engine = TD3Engine(
        gamma=cfg.gamma,
        tau=cfg.tau,
        observation_space=env.observation_space,
        action_space=env.action_space,
        actor=actor,
        critic_1=critic_1,
        critic_2=critic_2,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        policy_delay=cfg.policy_delay,
        target_policy_noise=cfg.target_noise,
        target_noise_clip=cfg.noise_clip,
        device=cfg.device,
    )

    replay_buf = SequenceBuffer(capacity=cfg.replay_buffer_size)

    ou_noise = OUNoise(
        action_dimension=action_dim,
        mu=0,
        theta=0.15,
        sigma=cfg.sigma_start,
        sigma_end=cfg.sigma_end,
        sigma_decay_epis=cfg.sigma_decay_episodes,
    )

    # --- Logging ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"trial_{trial.number}_{timestamp}"
    log_dir = f'out/runs/optuna/{study_name}/{run_name}'
    writer = SummaryWriter(log_dir)

    os.makedirs(log_dir, exist_ok=True)
    
    # Save the config and trial params
    params_path = os.path.join(log_dir, "trial_params.json")
    with open(params_path, "w") as f:
        json.dump(trial.params, f, indent=4)
        
    config_path = os.path.join(log_dir, "full_config.json")
    with open(config_path, "w") as f:
        f.write(cfg.to_json())

    try:
        td3_train(
            env=env,
            replay_buf=replay_buf,
            ou_noise=ou_noise,
            engine=engine,
            writer=writer,
            timestamp=timestamp,
            config=cfg,
            trial=trial
        )

        eval_ret, _eval_avg_action = engine.evaluate_policy(env, episodes=100)
        return eval_ret
    
    except optuna.TrialPruned:
        return -np.inf
    
    finally:
        env.close()
        writer.close()

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    study_name = f"twc_mcc_td3_bptt_optuna_{timestamp}"
    
    # Using TPE Sampler (common default) and MedianPruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Wait for 5 trials before pruning
        n_warmup_steps=30,   # Wait for 30 episodes (3 evals)
        interval_steps=10    # Prune every 10 episodes (1 eval)
    )
    study = optuna.create_study(
        direction="maximize", 
        sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(), 
        pruner=pruner,
        storage="sqlite:///db.sqlite3", # Save results to a DB
        study_name=study_name,
        load_if_exists=True # Resume from a previous study if name matches
    )

    # Wrapper for the objective function to pass the study_name
    # This is a clean way to get the study_name into the objective for logging
    obj_wrapper = lambda trial: objective(trial, study_name)

    # Run optimization
    try:
        study.optimize(
            obj_wrapper, 
            n_trials=20,
            gc_after_trial=True,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Optuna study interrupted by user.")

    # Print summary
    print(f"\nStudy '{study_name}' complete.")
    print(f"Best value (max eval return): {study.best_value:.2f}")
    print("Best params:")
    print(json.dumps(study.best_params, indent=4))
    
    # Save best params to a file
    best_params_path = os.path.join(f"out/runs/optuna/{study_name}", "best_params.json")
    os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Best params saved to {best_params_path}")

if __name__ == "__main__":
    main()
