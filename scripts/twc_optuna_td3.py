import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import itertools
import json
import gymnasium as gym
import numpy as np
import torch
import optuna
import optunahub
from datetime import datetime
from functools import partial
from td3 import TD3Engine
from utils import ReplayBuffer, OUNoise
from mlp import Critic
from twc import (
    build_twc,
    mcc_obs_encoder,
    mcc_obs_encoder_speed_weighted,
    twc_out_2_mcc_action,
    twc_out_2_mcc_action_tanh,
)

def make_env(seed, env_id="MountainCarContinuous-v0"):
    import gymnasium as gym
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

def objective(trial: optuna.Trial):
    # Fixed paramters
    MAX_STEPS          = trial.suggest_int("max_steps", 300_000, 300_000)
    EVAL_EVERY          = trial.suggest_int("eval_every", 10_000, 10_000)
    EVAL_EPIS           = trial.suggest_int("eval_epis", 10, 10)
    WARMUP_STEPS       = trial.suggest_int("warmup", 10_000, 10_000)
    TWC_INTERNAL_STEPS = trial.suggest_int("twc_in_steps", 3, 3)
    CRITIC_HID_LAYERS  = [400, 300]
    SIGMA_START, SIGMA_END, SIGMA_DECAY_EPIS = 0.20, 0.05, 100
    NUM_UPDATE_LOOPS = trial.suggest_categorical("num_update_loops", [2])
    POLICY_DELAY = trial.suggest_categorical("policy_delay", [2])
    DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameter search space
    GAMMA = trial.suggest_float("gamma", 0.95, 0.99)
    TAU   = trial.suggest_float("tau",   1e-3, 5e-3)
    ACTOR_LR  = trial.suggest_float("actor_lr", 1e-4, 5e-4, log=True)
    CRITIC_LR = trial.suggest_float("critic_lr", 1e-4, 5e-3, log=True)
    BATCH_SIZE = trial.suggest_categorical("batch_size", [64, 128])
    TARGET_NOISE = trial.suggest_float("target_policy_noise", 0.1, 0.3)
    NOISE_CLIP   = trial.suggest_float("target_noise_clip",   0.3, 0.6)

    # Seed per trial
    seed = 42 + trial.number
    np.random.seed(seed); torch.manual_seed(seed)
    env = make_env(seed)

    # Build models per trial to avoid cross-trial state leakage
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = build_twc(
        obs_encoder=mcc_obs_encoder_speed_weighted,
        action_decoder=twc_out_2_mcc_action_tanh,
        internal_steps=TWC_INTERNAL_STEPS,
        log_stats=False,
    )
    critic_1 = Critic(state_dim, action_dim, size=CRITIC_HID_LAYERS)
    critic_2 = Critic(state_dim, action_dim, size=CRITIC_HID_LAYERS)

    # Optimizers
    actor_opt = torch.optim.Adam(actor.parameters(),  lr=ACTOR_LR)
    critic_opt = torch.optim.Adam(
        itertools.chain(critic_1.parameters(), critic_2.parameters()),
        lr=CRITIC_LR,
    )

    engine = TD3Engine(
        gamma=GAMMA,
        tau=TAU,
        observation_space=env.observation_space,
        action_space=env.action_space,
        actor=actor,
        critic_1=critic_1,
        critic_2=critic_2,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        policy_delay=POLICY_DELAY,
        target_policy_noise=TARGET_NOISE,
        target_noise_clip=NOISE_CLIP,
        device=DEVICE,
    )

    replay_buf = ReplayBuffer(
        obs_dim=state_dim,
        act_dim=action_dim,
        size=100_000,
        keep=20_000,
    )

    ou_noise = OUNoise(
        action_dimension=action_dim,
        mu=0,
        theta=0.15,
        sigma=SIGMA_START,
        sigma_end=SIGMA_END,
        sigma_decay_epis=SIGMA_DECAY_EPIS,
    )

    obs, _ = env.reset()
    best_ret = -float("inf")
    total_steps = 0
    ep = 0
    while total_steps < MAX_STEPS:
        if total_steps < WARMUP_STEPS:
            action = env.action_space.sample()
        else:
            action = engine.get_action(obs, action_noise=ou_noise.noise)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        replay_buf.store(obs, action, reward, next_obs, terminated, truncated)
        obs = next_obs
        total_steps += 1

        if total_steps > WARMUP_STEPS and replay_buf.size >= BATCH_SIZE:
            for _ in range(NUM_UPDATE_LOOPS):
                batch = replay_buf.sample(BATCH_SIZE, DEVICE)
                _actor_loss, _critic_loss = engine.update_step(batch)

        if terminated or truncated:
            obs, _ = env.reset()
            ep += 1
            ou_noise.update_sigma(ep)

        if total_steps % EVAL_EVERY == 0:
            eval_ret, _eval_avg_action = engine.evaluate_policy(env, episodes=EVAL_EPIS)
            best_ret = max(best_ret, eval_ret)
            if eval_ret > best_ret:
                torch.save(engine.actor.state_dict(), f"out/models/twc_td3_opt_actor_best_{trial.number}.pth")
            trial.report(eval_ret, total_steps)
            if trial.should_prune():
                env.close()
                raise optuna.TrialPruned()

    eval_ret, _eval_avg_action = engine.evaluate_policy(env, episodes=100)
    env.close()
    return eval_ret

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Single-objective study (maximize return), with optional pruning
    sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler()
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", 
                                sampler=sampler, 
                                pruner=pruner,
                                storage="sqlite:///db.sqlite3",
                                study_name=f"twc_td3_optuna_{timestamp}")

    # Run optimization
    study.optimize(objective, n_trials=20, gc_after_trial=True, show_progress_bar=True)

    # Print summary
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")

if __name__ == "__main__":
    main()
