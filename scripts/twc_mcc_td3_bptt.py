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

def main():
    cfg = TD3Config()
    # --- Set Fixed Parameters ---
    cfg.use_bptt = True 
    cfg.max_episode = 300
    cfg.max_time_steps = 999
    cfg.warmup_steps = 10_000
    cfg.eval_interval_episodes = 10
    cfg.eval_episodes = 10
    cfg.sequence_length = 10
    cfg.burn_in_length = 4
    cfg.num_update_loops = 2
    # Model arch params
    cfg.twc_internal_steps = 1
    cfg.critic_hidden_layers = [400, 300]

    # --- Set Tunable Hyperparameters ---
    cfg.batch_size = 256
    cfg.actor_lr = 0.0008
    cfg.critic_lr = 0.0005
    cfg.gamma = 0.9844956923106524
    cfg.tau = 0.0040931345282959174
    cfg.policy_delay = 2
    
    # Noise params
    cfg.target_noise = 0.2
    cfg.noise_clip = 0.3599783289890277
    cfg.sigma_start = 0.29976116138156494
    cfg.sigma_end = 0.05
    cfg.sigma_decay_episodes = 150.0
    cfg.replay_buffer_size = 100_000
    # Seed per trial
    seed = 42
    np.random.seed(seed); torch.manual_seed(seed)
    env = make_env(seed)

    # Build models per trial to avoid cross-trial state leakage
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = build_twc(
        obs_encoder=mcc_obs_encoder,
        action_decoder=twc_out_2_mcc_action,
        internal_steps=cfg.twc_internal_steps,
        log_stats=False,
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
    run_name = f"twc_td3_bptt_{timestamp}"
    log_dir = f'out/runs/td3/{run_name}'
    writer = SummaryWriter(log_dir)

    os.makedirs(log_dir, exist_ok=True)
        
    config_path = os.path.join(log_dir, "full_config.json")
    with open(config_path, "w") as f:
        f.write(cfg.to_json())

    td3_train(
            env=env,
            replay_buf=replay_buf,
            ou_noise=ou_noise,
            engine=engine,
            writer=writer,
            timestamp=timestamp,
            config=cfg,
            trial=None
        )
    
    # save final models
    prefix = "td3_actor_final"
    model_path = os.path.join(writer.log_dir, f"{prefix}_{timestamp}.pth")
    torch.save(engine.actor.state_dict(), model_path)

if __name__ == "__main__":
    main()