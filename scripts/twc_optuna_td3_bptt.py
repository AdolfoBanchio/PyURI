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
from optuna.samplers import TPESampler
from optuna.trial import TrialState

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
    cfg.max_train_steps = 250_000
    cfg.max_time_steps = 999
    cfg.warmup_steps = 10_000
    cfg.eval_interval_episodes = 10
    cfg.eval_episodes = 10
    cfg.model_prefix = "twc_td3_OU_actor"
    
    # Models fixed params
    cfg.critic_hidden_layers = [400, 300]
    cfg.twc_internal_steps = 1
    cfg.rnd_init = True
    cfg.use_v2 = True
    
    # BPTT related
    cfg.use_bptt = True 
    cfg.sequence_length = 8
    cfg.burn_in_length = 4  
    cfg.batch_size = 256
    cfg.num_update_loops = 2 
    cfg.policy_delay = 2 # , policy is updated every 2 update steps
    
    # --- Set Tunable Hyperparameters ---
    # Centered around best-known config (20251202 success) with ±30–40% ranges
    cfg.actor_lr = trial.suggest_float("actor_lr", 1.5e-4, 3.0e-4, log=True)      # best ~2.24e-4
    cfg.critic_lr = trial.suggest_float("critic_lr", 1.2e-4, 3.0e-4, log=True)    # best ~1.83e-4
    cfg.gamma = trial.suggest_float("gamma", 0.978, 0.990)                        # best ~0.98235
    cfg.tau   = trial.suggest_float("tau", 5e-3, 1.2e-2)                          # best ~0.00769
    cfg.target_noise = trial.suggest_float("target_noise", 0.20, 0.36)            # best ~0.284
    cfg.noise_clip   = trial.suggest_float("noise_clip",   0.25, 0.40)            # best ~0.318
    cfg.ou_sigma_init = trial.suggest_float("sigma_start", 0.30, 0.45)            # best ~0.396
    cfg.ou_sigma_end = trial.suggest_float("sigma_end", 0.05, 0.12)               # best ~0.082

    if cfg.use_v2:
        # Narrow around the successful run
        cfg.steepness_fire = trial.suggest_float("steep_fire", 12.0, 16.0)       # best 14.4345
        cfg.steepness_gj = trial.suggest_float("steep_gj", 6.0, 9.0)             # best 7.1332
        cfg.steepness_input = trial.suggest_float("steep_input", 4.0, 6.0)       # best 4.9847
        cfg.input_thresh = trial.suggest_float("input_thresh", 8e-4, 2e-3, log=True)  # best 0.00124
        cfg.leaky_slope = trial.suggest_float("leaky_slope", 0.015, 0.035)       # best 0.0231
        v2_params = {
            'steepness_fire': cfg.steepness_fire,
            'steepness_gj': cfg.steepness_gj,
            'steepness_input': cfg.steepness_input,
            'input_thresh': cfg.input_thresh,
            'leaky_slope': cfg.leaky_slope,
            }
        
    # Seed per trial
    seed = 42 + trial.number
    cfg.seed = seed
    
    np.random.seed(seed)
    torch.manual_seed(seed)
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
        **v2_params
    )
    
    print(f"Actor of trial {trial.number}: ")
    print(actor.state_dict())
    
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

    # Exploration noise 
    noise = OUNoise(size=env.action_space.shape,
                       mu=0.0,
                       theta=0.15,
                       sigma_init=cfg.ou_sigma_init,
                       sigma_min=cfg.ou_sigma_end,
                       decay_steps=cfg.max_train_steps * 0.7,   # FIXED
                       seed=cfg.seed
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
        best_ret, best_model_path = td3_train(
            env=env,
            replay_buf=replay_buf,
            engine=engine,
            writer=writer,
            timestamp=timestamp,
            config=cfg,
            trial=trial,
            OUNoise=noise,
        )

        # Evaluate the saved best checkpoint (greedy, multi-episode)
        if best_model_path is not None:
            state_dict = torch.load(best_model_path, map_location=engine.device)
            engine.actor.load_state_dict(state_dict)
        eval_ret, _eval_avg_action = engine.evaluate_policy(env, episodes=100)
        return eval_ret
    
    except optuna.TrialPruned:
        return -np.inf
    
    finally:
        env.close()
        writer.close()

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Keep a stable study name so multiple workers can attach to the same DB
    study_name = "twc_mcc_td3_bptt_optuna_unconstrained_decay"

    # Using TPE Sampler (common default) and MedianPruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=4,   # let a few seeds run end-to-end before pruning
        n_warmup_steps=8,     # ~first 8 evals (~80 episodes) kept (good runs improve ~70k steps)
        interval_steps=2      # check every ~20 episodes (every other eval)
    )

    sampler = TPESampler(
        multivariate=True,      # modela dependencias entre hparams
        group=True,             # agrupa parámetros relacionados
    )
    study = optuna.create_study(
        direction="maximize", 
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:///db.sqlite3", # Shared DB for parallel workers
        study_name=study_name,
        load_if_exists=True # Resume/attach if already running
    )

    # Only enqueue the baseline once (first worker only)
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if len(study.trials) == 0:
        baseline = {
            "actor_lr": 2.239407231090426e-4,
            "critic_lr": 1.828306017572226e-4,
            "gamma": 0.9823522271023871,
            "tau": 0.007693135327059323,
            "target_noise": 0.28415959581368067,
            "noise_clip": 0.31789035300173857,
            "sigma_start": 0.39609107327435644,
            "sigma_end": 0.08244881107627974,
            "steep_fire": 14.4345331,
            "steep_gj": 7.1331877,
            "steep_input": 4.9846608,
            "input_thresh": 0.0012398,
            "leaky_slope": 0.0231012,
        }
        study.enqueue_trial(baseline)
    elif len(completed_trials) > 0:
        best_params = study.best_trial.params
        print("\nEnqueuing previous best params for re-evaluation...")
        study.enqueue_trial(best_params)


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

    # Print summary (only if we have completed trials)
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if completed_trials:
        print(f"\nStudy '{study_name}' complete.")
        print(f"Best value (max eval return): {study.best_value:.2f}")
        print("Best params:")
        print(json.dumps(study.best_params, indent=4))
        
        # Save best params to a file
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
