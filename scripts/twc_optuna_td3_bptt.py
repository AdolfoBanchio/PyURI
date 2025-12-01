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
    cfg.max_train_steps = 300_000
    cfg.max_time_steps = 999
    cfg.warmup_steps = 10_000
    cfg.eval_interval_episodes = 10
    cfg.eval_episodes = 10
    cfg.model_prefix = "twc_td3_OU_actor"
    
    # Models fixed params
    cfg.critic_hidden_layers = [400, 300]
    cfg.twc_internal_steps = 1
    cfg.rnd_init = True
    cfg.use_v2 = False
    
    # BPTT related
    cfg.use_bptt = True 
    cfg.sequence_length = 9
    cfg.burn_in_length = 5   
    cfg.batch_size = 256
    cfg.num_update_loops = 2 
    cfg.policy_delay = 2 # , policy is updated every 2 update steps
    
    # --- Set Tunable Hyperparameters ---
    cfg.actor_lr = trial.suggest_float("actor_lr", 1e-5, 2e-4, log=True)
    cfg.critic_lr = trial.suggest_float("critic_lr", 1e-4, 1e-3, log=True)

    cfg.gamma = trial.suggest_float("gamma", 0.985, 0.997)
    cfg.tau   = trial.suggest_float("tau", 5e-4, 1e-2, log=True)

    cfg.target_noise = trial.suggest_float("target_noise", 0.1, 0.4)
    cfg.noise_clip   = trial.suggest_float("noise_clip",   0.2, 0.6)
    cfg.ou_sigma_init = trial.suggest_float("sigma_start", 0.35, 0.5)
    cfg.ou_sigma_end = trial.suggest_float("sigma_end", 0.03, 0.08)

    v2_params = {}
    if cfg.use_v2:
        # Surrogate gradients TWC hyperparameters
        # Trial 0: 14.43 -> Rango [12, 16]
        cfg.steepness_fire = trial.suggest_float("steep_fire", 12.0, 16.0)
        # Trial 0: 7.13 -> Rango [6, 9]
        cfg.steepness_gj = trial.suggest_float("steep_gj", 6.0, 9.0)
        # Trial 0: 4.98 -> Rango [4, 6]
        cfg.steepness_input = trial.suggest_float("steep_input", 4.0, 6.0)
        # Trial 0: 0.0012 -> Rango [0.0008, 0.002]
        cfg.input_thresh = trial.suggest_float("input_thresh", 8e-4, 2e-3, log=True)
        # Trial 0: 0.023 -> Rango [0.015, 0.035]
        cfg.leaky_slope = trial.suggest_float("leaky_slope", 0.015, 0.035)
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
        td3_train(
            env=env,
            replay_buf=replay_buf,
            engine=engine,
            writer=writer,
            timestamp=timestamp,
            config=cfg,
            trial=trial,
            OUNoise=noise,
            use_PER=False,
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
    #study_name = f"twc_mcc_td3_bptt_optuna_{timestamp}"
    study_name = "twc_mcc_td3_bptt_optuna_20251117_092047"

    # Using TPE Sampler (common default) and MedianPruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Wait for 5 trials before pruning
        n_warmup_steps=10,   # Wait for 30 episodes (3 evals)
        interval_steps=3    # Prune every 10 episodes (1 eval)
    )

    sampler = TPESampler(
        multivariate=True,      # modela dependencias entre hparams
        group=True,             # agrupa parámetros relacionados
    )
    study = optuna.create_study(
        direction="maximize", 
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:///db.sqlite3", # Save results to a DB
        study_name=study_name,
        load_if_exists=True # Resume from a previous study if name matches
    )


    if len(study.trials) > 0:
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

    # Print summary
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

if __name__ == "__main__":
    main()
