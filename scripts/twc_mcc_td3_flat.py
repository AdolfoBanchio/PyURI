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
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import SequenceBuffer
from mlp import TwinCritic
from fiuri import build_fiuri_twc, build_fiuri_twc_v2
from td3_flat import TD3Engine, TD3Config, td3_train


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
    parser.add_argument(
        "--use-sg",
        action="store_true",
        help="Enable surrogate gradients (SG)"
    )
    return parser.parse_args()


def main(cfg: TD3Config, use_sg=False):
    # Seed per trial
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_env(seed)
    
    # Build models per trial to avoid cross-trial state leakage
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if use_sg:
        actor = build_fiuri_twc_v2(steepness_fire=cfg.steepness_fire,
                                   steepness_gj=cfg.steepness_gj,
                                   steepness_input=cfg.steepness_input,
                                   input_thresh=cfg.input_thresh)
        dir_name = "td3_flat_twc"
    else:
        dir_name = "td3_flat_noSG_twc"
        actor = build_fiuri_twc()
        
    critic = TwinCritic(state_dim=state_dim, action_dim=action_dim)
    
    # Optimizers
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    actor_opt = torch.optim.Adam(actor.parameters(),  lr=cfg.actor_lr)
    
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

    # --- Logging ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"twc_mcc_flat_twc_{timestamp}"
    log_dir = f'out/runs/{dir_name}/{run_name}'
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
        )


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config_path)
    use_sg = args.use_sg
    print(config_path)
    cfg = TD3Config()
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data =  json.load(f)
        cfg = cfg.load(config_data)

    print(cfg)
    main(cfg, use_sg)