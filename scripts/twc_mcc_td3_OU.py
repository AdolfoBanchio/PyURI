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
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from functools import partial
from td3 import TD3Engine, TD3Config, td3_train
from utils import OUNoise, SequenceBuffer
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a MCC agent using TD3 and TWC architecture"
    )
    parser.add_argument("config_path", type=str, help="Path to the TD3 Config json")

    return parser.parse_args()

import numpy as np
import torch


def main(cfg: TD3Config):
    # V2 twc hyperparameters
    v2_params = {}
    if cfg.use_v2:
        v2_params = {
            'steepness_fire': cfg.steepness_fire,
            'steepness_gj': cfg.steepness_gj,
            'steepness_input': cfg.steepness_input,
            'input_thresh': cfg.input_thresh,
            'leaky_slope': cfg.leaky_slope,
            }
    # Seed per trial
    seed = cfg.seed
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


    noise = OUNoise(size=env.action_space.shape,
                       mu=0.0,
                       theta=0.15,
                       sigma_init=cfg.ou_sigma_init,
                       sigma_min=cfg.ou_sigma_end,
                       decay_steps=cfg.max_train_steps * 0.7,   # 300k
                       dt=1.0,
                       seed=cfg.seed
                       )

    # --- Logging ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"twc_mcc_V{cfg.use_v2}_OU_{timestamp}"
    log_dir = f'out/runs/td3_OU/{run_name}'
    writer = SummaryWriter(log_dir)

    os.makedirs(log_dir, exist_ok=True)
        
    config_path = os.path.join(log_dir, "full_config.json")
    with open(config_path, "w") as f:
        f.write(cfg.to_json())
    
    # Trains, saves best and final models. 
    td3_train(
            env=env,
            replay_buf=replay_buf,
            engine=engine,
            writer=writer,
            timestamp=timestamp,
            config=cfg,
            OUNoise=noise,
            use_PER=False,
        )


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config_path)
    print(config_path)
    cfg = TD3Config()
    if config_path.exists:
        with open(config_path, 'r') as f:
            config_data =  json.load(f)
        cfg = cfg.load(config_data)

    print(cfg)
    main(cfg)