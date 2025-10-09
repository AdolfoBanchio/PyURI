# dqn_twc_mountaincar_clean.py
import math
import random
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

# --- imports for your model
from utils.twc_builder import build_twc
from utils.twc_io_wrapper import mountaincar_pair_encoder

# ------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------
@dataclass
class DQNConfig:
    env_id: str = "MountainCar-v0"
    total_steps: int = 200_000
    train_start: int = 5_000
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    lr: float = 3e-4
    target_update_every: int = 1_000
    tau: float = 1.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000
    max_grad_norm: float = 1.0
    seed: int = 123
    render: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every_episodes: int = 20
    double_dqn: bool = True

# ------------------------------------------------------
# UTILITIES
# ------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def linear_eps(step, start, end, decay_steps):
    if step >= decay_steps:
        return end
    t = step / float(decay_steps)
    return start + (end - start) * t

def soft_update_(online: nn.Module, target: nn.Module, tau: float):
    if tau >= 1.0:
        target.load_state_dict(online.state_dict())
    else:
        with torch.no_grad():
            for tp, op in zip(target.parameters(), online.parameters()):
                tp.data.mul_(1 - tau).add_(op.data, alpha=tau)

# ------------------------------------------------------
# REPLAY BUFFER
# ------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity,), dtype=np.int64)
        self.rew_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs_buf[idxs], device=self.device)
        act = torch.as_tensor(self.act_buf[idxs], device=self.device)
        rew = torch.as_tensor(self.rew_buf[idxs], device=self.device)
        next_obs = torch.as_tensor(self.next_obs_buf[idxs], device=self.device)
        done = torch.as_tensor(self.done_buf[idxs], device=self.device)
        return obs, act, rew, next_obs, done

# ------------------------------------------------------
# NETWORK WITH TWC
# ------------------------------------------------------
class QNetTWC(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, device: torch.device):
        super().__init__()
        self.device = device
        self.twc = build_twc(action_decoder=mountaincar_pair_encoder(), use_json_w=True).to(device)
        with torch.no_grad():
            dummy = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
            n_out = self.twc(dummy).shape[-1]
            self.twc.reset()
        self.head = nn.Linear(n_out, n_actions)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.constant_(self.head.bias, 0.0)

    @torch.no_grad()
    def act_stateful(self, obs: np.ndarray, eps: float) -> int:
        if np.random.rand() < eps:
            return np.random.randint(self.head.out_features)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.forward_stateful(obs_t)
        return int(torch.argmax(q, dim=-1).item())

    @torch.no_grad()
    def forward_stateful(self, obs_t: torch.Tensor) -> torch.Tensor:
        y = self.twc(obs_t)
        q = self.head(y)
        self.twc.detach()
        return q

    def forward_stateless(self, obs_b: torch.Tensor) -> torch.Tensor:
        self.twc.reset()
        y = self.twc(obs_b)
        q = self.head(y)
        return q

# ------------------------------------------------------
# TRAINER
# ------------------------------------------------------
def train_dqn(cfg: DQNConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    env = gym.make(cfg.env_id, render_mode="human" if cfg.render else None)
    obs, _ = env.reset(seed=cfg.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n

    online = QNetTWC(obs_dim, n_actions, device=device)
    target = QNetTWC(obs_dim, n_actions, device=device)
    soft_update_(online, target, tau=1.0)

    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_size, obs_dim, device=device)

    ep_return, ep_len, episodes = 0.0, 0, 0
    recent_returns = deque(maxlen=cfg.log_every_episodes)
    pbar = trange(cfg.total_steps, desc="Training DQN", ncols=100)

    for global_step in range(cfg.total_steps):
        eps = linear_eps(global_step, cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps)
        action = online.act_stateful(obs, eps)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, next_obs, done)

        ep_return += reward
        ep_len += 1
        obs = next_obs

        if buffer.size >= cfg.train_start:
            obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(cfg.batch_size)
            q_s_all = online.forward_stateless(obs_b)
            q_s = q_s_all.gather(1, act_b.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                if cfg.double_dqn:
                    next_actions = online.forward_stateless(next_obs_b).argmax(1, keepdim=True)
                    q_next = target.forward_stateless(next_obs_b).gather(1, next_actions).squeeze(1)
                else:
                    q_next = target.forward_stateless(next_obs_b).max(1).values

                target_q = rew_b + cfg.gamma * (1 - done_b) * q_next

            loss = F.smooth_l1_loss(q_s, target_q)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), cfg.max_grad_norm)
            opt.step()

            if global_step % cfg.target_update_every == 0:
                soft_update_(online, target, cfg.tau)

        # --- Episode end
        if done:
            episodes += 1
            recent_returns.append(ep_return)
            obs, _ = env.reset()
            online.twc.reset()
            target.twc.reset()
            ep_return, ep_len = 0.0, 0

            # Print every few episodes
            if episodes % cfg.log_every_episodes == 0:
                avg_ret = np.mean(recent_returns)
                print(f"[Ep {episodes:4d}] Step {global_step:6d} | "
                      f"AvgReturn: {avg_ret:6.2f} | Eps: {eps:5.2f} | Buffer: {buffer.size}")

    env.close()
    pbar.close()
    print("Training finished.")

# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
if __name__ == "__main__":
    cfg = DQNConfig(
        total_steps=200_000,
        log_every_episodes=20,  # print every 20 episodes
        render=True,
    )
    train_dqn(cfg)
