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
import ast
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
from dataclasses import dataclass, asdict
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from functools import partial
from td3 import td3_train
from utils import OUNoise, SequenceBuffer
from mlp import BestCritic
from fiuri import PyUriTwc_V2, build_fiuri_twc_v2

@dataclass
class TD3Config:
    # --- Training loop ---
    max_train_steps: int = 300_000
    warmup_steps: int = 10_000
    batch_size: int = 256  # Standard stable batch size for MCC
    num_update_loops: int = 1 # Updates per step (Standard is 1)
    update_every: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # --- Evaluation ---
    eval_interval_episodes: int = 10
    eval_episodes: int = 10

    # --- BPTT options ---
    sequence_length: int = 8
    burn_in_length: int = 4

    # Hyperparameters
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5
    
    exp_noise: float = 0.1 # most common fixed std in TD3 algorithms
    # OU Noise Parameters
    ou_theta: float = 0.15
    ou_sigma_init: float = 0.5 # Higher initial noise for exploration
    ou_sigma_end: float = 0.1
    ou_sigma_decay_steps: int = 100_000 # Decays over first 1/3 of training
    
    # SG version hyperparameters
    steepness_fire: float = 14.0
    steepness_gj: float = 7.0
    steepness_input: float = 5.0
    input_thresh: float = 0.001
    leaky_slope: float = 0.02

    critic_hidden_layers: int = 256    
    replay_buffer_size: int = 100_000
    
    model_prefix: str = "td3_flat_actor"
    
    def to_json(self) -> str:
        d = asdict(self)
        d["critic_hidden_layers"] = str(self.critic_hidden_layers)
        return json.dumps(d, indent=4)

    def load(self, json_data):
        data = json.loads(json_data) if isinstance(json_data, str) else dict(json_data)
        
        # Handle tuple fields
        layers = data.get("critic_hidden_layers")
        if isinstance(layers, str):
            try:
                layers = json.loads(layers)
            except:
                layers = ast.literal_eval(layers)
        if isinstance(layers, (list, tuple)):
            data["critic_hidden_layers"] = tuple(layers)

        for field in self.__dataclass_fields__:
            if field in data:
                setattr(self, field, data[field])
        return self
    
class TD3Engine():
    """  
    TODO: UPDATE DOCSTRING
    
    """
    def __init__(self,
                 gamma: float,
                 tau: float,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 actor: PyUriTwc_V2,
                 critic_1: nn.Module,
                 critic_2: nn.Module,
                 actor_optimizer: torch.optim.Optimizer,
                 critic_optimizer: torch.optim.Optimizer,
                 policy_delay: int = 2,
                 target_policy_noise: float = 0.2,
                 target_noise_clip: float = 0.5,
                 device: torch.device = torch.device("cpu")):
        
        self.gamma = gamma
        self.tau = tau
        self.obs_space = observation_space
        self.act_space = action_space
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.actor_target = deepcopy(actor)
        self.critic_1_target = deepcopy(critic_1)
        self.critic_2_target = deepcopy(critic_2)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer  # should include both critics' params
        self.device = device

        self.actor.to(device)
        self.critic_1.to(device)
        self.critic_2.to(device)
        self.actor_target.to(device)
        self.critic_1_target.to(device)
        self.critic_2_target.to(device)

        self.policy_delay = int(policy_delay)
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)

        # cache bounds as tensors
        self.action_low  = torch.as_tensor(self.act_space.low,  device=self.device, dtype=torch.float32)
        self.action_high = torch.as_tensor(self.act_space.high, device=self.device, dtype=torch.float32)

        self.total_updates = 0  # for delayed actor

        # compile actors
        """ self.actor.compile(mode="default")
        self.actor_target.compile(mode="default")
        self.critic_1.compile(mode="default")
        self.critic_2.compile(mode="default")
        self.critic_1_target.compile(mode="default")
        self.critic_2_target.compile(mode="default") """
    
    @torch.no_grad()
    def soft_update(self, net: nn.Module, target_net: nn.Module):
            for p, tp in zip(net.parameters(), target_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
                
    def get_action(self, state):
        # Shape: (1, ObsDim)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            a = self.actor(s) # Expected (1, ActionDim)

        # Clamp using broadcasted shapes, then squeeze to (ActionDim,)
        return torch.clamp(a, self.action_low, self.action_high).squeeze(0)
    
    def set_eval(self):
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.actor_target.eval()
        self.critic_1_target.eval()
        self.critic_2_target.eval()
    
    def set_train(self):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.actor_target.train()
        self.critic_1_target.train()
        self.critic_2_target.train()

    def _detach_state_tuple(self, state_tuple):
        """Helper to detach (E, O) tuple from TWC_V2"""
        return (state_tuple[0].detach(), state_tuple[1].detach())
    
    def update_step_bptt(self, 
                         batch: dict, 
                         burn_in: int,):
        obs = batch['obs'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        done = batch['done'].to(self.device)
        
        B = obs.shape[0]

        # 1. Burn-in
        with torch.no_grad():
            init_E, init_O = self.actor.get_initial_state(B, self.device)
            if burn_in > 0:
                _, (h_E, h_O) = self.actor.forward_bptt(obs[:, :burn_in], (init_E, init_O))
                _, (h_E_t, h_O_t) = self.actor_target.forward_bptt(next_obs[:, :burn_in], (init_E, init_O))
            else:
                h_E, h_O = init_E, init_O
                h_E_t, h_O_t = init_E, init_O
            
        # Training Sequence Slices
        obs_t = obs[:, burn_in:]
        next_obs_t = next_obs[:, burn_in:]
        act_t = action[:, burn_in:]
        rew_t = reward[:, burn_in:].unsqueeze(-1)
        done_t = done[:, burn_in:].unsqueeze(-1)

        # 2. Critic Update
        with torch.no_grad():
            # Get next action from target actor
            next_act, _ = self.actor_target.forward_bptt(next_obs_t, (h_E_t, h_O_t))
            
            # Add noise
            noise = (torch.randn_like(next_act) * self.target_policy_noise).clamp(-self.target_noise_clip, self.target_noise_clip)
            next_act = (next_act + noise).clamp(self.action_low.unsqueeze(1), self.action_high.unsqueeze(1))            
            # Target Q
            q1_t = self.critic_1_target(next_obs_t, next_act)
            q2_t = self.critic_2_target(next_obs_t, next_act)
            min_q = torch.min(q1_t, q2_t)
            
            target_q = rew_t + (1 - done_t) * self.gamma * min_q

        q1 = self.critic_1(obs_t, act_t)
        q2 = self.critic_2(obs_t, act_t)
        
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
            1.0,
        )
        self.critic_optimizer.step()

        # 3. Actor Update
        actor_loss_val = 0.0
        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:

            h_E_detached = h_E.detach()
            h_O_detached = h_O.detach()

            pi_act, _ = self.actor.forward_bptt(obs_t, (h_E_detached, h_O_detached))
            actor_loss = -self.critic_1(obs_t, pi_act).mean()
            
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Soft update targets
            self.soft_update(net=self.actor,   target_net=self.actor_target)
            self.soft_update(net=self.critic_1, target_net=self.critic_1_target)
            self.soft_update(net=self.critic_2, target_net=self.critic_2_target)
            
            actor_loss_val = actor_loss.item()
            
        return actor_loss_val, critic_loss.item(), 
    
    @torch.no_grad()
    def evaluate_policy(self, env, episodes=10):
        self.actor.eval()
        total = 0.0
        eval_actions = []
        for _ in range(episodes):
            obs, _ = env.reset()
            self.actor.reset()
            done = False
            while not done:
                a = self.get_action(obs)
                a_np = a.detach().cpu().numpy()
                eval_actions.append(a_np)
                obs, r, terminated, truncated, _ = env.step(a_np)
                total += r
                done = terminated or truncated
        self.actor.train()
        return (total / episodes), np.mean(eval_actions)
        
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


def main(cfg: TD3Config):
    # Seed per trial
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = cfg.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_env(seed)
    
    # Build models per trial to avoid cross-trial state leakage
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = build_fiuri_twc_v2(
        steepness_gj=cfg.steepness_gj,
        steepness_fire=cfg.steepness_fire,
        steepness_input=cfg.steepness_input,
        input_thresh=cfg.input_thresh,
        leaky_slope=cfg.leaky_slope,   
    )
    
    critic_1 = BestCritic(state_dim=state_dim, action_dim=action_dim)
    critic_2 = BestCritic(state_dim=state_dim, action_dim=action_dim)
    
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
    run_name = f"twc_mcc_flat_twc_{timestamp}"
    log_dir = f'out/runs/td3_flat_twc/{run_name}'
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
        )


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config_path)
    print(config_path)
    cfg = TD3Config()
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data =  json.load(f)
        cfg = cfg.load(config_data)

    print(cfg)
    main(cfg)