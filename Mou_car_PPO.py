import math
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.twc_builder import build_twc
from utils.twc_io_wrapper import mountaincar_pair_encoder


# -------- Utility: stable atanh for log-prob correction --------
def atanh_safe(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a = a.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(a) - torch.log1p(-a))


LOG_2PI = 1.8378770664093453


def policy_params_from_outstate(out_state: torch.Tensor,
                                *, gain: float = 1.0,
                                min_log_std: float = -5.0,
                                max_log_std: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map TWC output (B,2) ordered [REV, FWD] into Gaussian params for a 1-D action.
    mean = gain * (FWD - REV); log_std bounded via a smooth squashing of (FWD+REV).
    Returns (mean, log_std) each (B,1).
    """
    assert out_state.ndim == 2 and out_state.shape[1] == 2, "expected (B,2)"
    rev, fwd = out_state[:, 0], out_state[:, 1]
    mean = gain * (fwd - rev)
    raw = fwd + rev
    log_std = min_log_std + 0.5 * (torch.tanh(raw) + 1.0) * (max_log_std - min_log_std)
    log_std = torch.clamp(log_std, min_log_std, max_log_std)
    return mean.unsqueeze(-1), log_std.unsqueeze(-1)


def tanh_gaussian_logp(mean: torch.Tensor, log_std: torch.Tensor, action: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Log-probability under tanh-squashed Gaussian: a = tanh(u), u ~ N(mean, std).
    Includes change-of-variables correction term.
    Shapes: mean/log_std/action are (B,1). Returns (B,1).
    """
    std = log_std.exp()
    u = atanh_safe(action, eps=eps)
    var = std * std
    logp_base = -0.5 * (((u - mean) ** 2) / (var + eps) + 2.0 * log_std + LOG_2PI)
    squash = torch.log(1.0 - action * action + eps)
    return logp_base - squash


@torch.no_grad()
def tanh_gaussian_sample(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    std = log_std.exp()
    u = mean + std * torch.randn_like(std)
    return torch.tanh(u)


# -------- Rollout buffer and GAE --------
class Rollout:
    """
    Simple rollout buffer for on-policy PPO.
    Stores obs, actions, log-probs, rewards, dones, and values.
    """
    def __init__(self, T: int, obs_dim: int, device: torch.device):
        self.obs = torch.zeros(T, obs_dim, device=device)
        self.actions = torch.zeros(T, 1, device=device)
        self.logp = torch.zeros(T, 1, device=device)
        self.rew = torch.zeros(T, 1, device=device)
        self.done = torch.zeros(T, 1, device=device)
        self.val = torch.zeros(T, 1, device=device)
        self.ptr = 0

    def add(self, obs, action, logp, reward, done, value):
        i = self.ptr
        self.obs[i].copy_(obs.squeeze(0))
        self.actions[i].copy_(action)
        self.logp[i].copy_(logp)
        self.rew[i, 0] = reward
        self.done[i, 0] = float(done)
        self.val[i].copy_(value)
        self.ptr += 1


@torch.no_grad()
def compute_gae(rew: torch.Tensor,
                val: torch.Tensor,
                done: torch.Tensor,
                last_value: torch.Tensor,
                gamma: float,
                lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation.
    Inputs: rew/val/done (T,1), last_value (1,1).
    Returns: advantages and returns (each T,1).
    """
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    next_value = last_value
    next_nonterm = torch.ones(1, 1, device=rew.device)
    lastgaelam = torch.zeros(1, 1, device=rew.device)

    for t in reversed(range(T)):
        nonterm = 1.0 - done[t:t+1]
        delta = rew[t:t+1] + gamma * next_value * next_nonterm - val[t:t+1]
        lastgaelam = delta + gamma * lam * next_nonterm * lastgaelam
        adv[t:t+1] = lastgaelam
        next_value = val[t:t+1]
        next_nonterm = nonterm

    ret = adv + val
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


# -------- PPO Agent --------
@dataclass
class PPOConfig:
    env_id: str = "MountainCarContinuous-v0"
    total_timesteps: int = 50_000
    rollout_steps: int = 1024
    update_epochs: int = 10
    minibatch_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    seed: int = 42
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ValueHead(nn.Module):
    """
    Small value head mapping TWC output (2,) to scalar value.
    """
    def __init__(self, in_dim: int = 2):
        super().__init__()
        self.v = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(x)


class TwcPPOAgent:
    """
    PPO agent using the biologically inspired TWC network for policy and value.

    - Actor/Critic are separate TWC instances to avoid unwanted gradient sharing.
    - TWC takes raw observations (B,2) and uses the provided encoder internally.
    - Actions are sampled from a tanh-squashed Gaussian built from TWC outputs.
    """
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.device = cfg.device
        self.env = gym.make(cfg.env_id)
        try:
            self.env.reset(seed=cfg.seed)
            self.env.action_space.seed(cfg.seed)
            self.env.observation_space.seed(cfg.seed)
        except Exception:
            pass

        self.obs_space = self.env.observation_space
        self.act_space = self.env.action_space

        # Build actor and critic TWC networks; both use the same observation encoder.
        self.actor = build_twc(action_decoder=mountaincar_pair_encoder(), use_json_w=True).to(self.device)
        self.critic = build_twc(action_decoder=mountaincar_pair_encoder(), use_json_w=True).to(self.device)
        self.value_head = ValueHead(in_dim=2).to(self.device)

        params = list(self.actor.parameters()) + list(self.critic.parameters()) + list(self.value_head.parameters())
        self.optimizer = optim.Adam(params, lr=cfg.lr)

    def rollout(self, steps: int):
        buf = Rollout(steps, obs_dim=self.obs_space.shape[0], device=self.device)
        obs, _ = self.env.reset()

        for t in range(steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Reset recurrent-like state per step for stability
            self.actor.reset(); self.critic.reset()

            # Policy forward and sample action
            with torch.no_grad():
                out_act = self.actor.forward(obs_t)        # (1,2)
            mean, log_std = policy_params_from_outstate(out_act)
            a = tanh_gaussian_sample(mean, log_std)        # (1,1)
            logp = tanh_gaussian_logp(mean, log_std, a)    # (1,1)

            # Value estimate
            with torch.no_grad():
                out_crit = self.critic.forward(obs_t)      # (1,2)
            v = self.value_head(out_crit)                  # (1,1)

            # Step environment
            act_np = a.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = self.env.step(act_np)
            done = terminated or truncated

            buf.add(obs_t, a.squeeze(0), logp.squeeze(0), reward, done, v.squeeze(0))
            obs = next_obs

            if done:
                obs, _ = self.env.reset()
                self.actor.reset(); self.critic.reset()

        # Bootstrap value at end of rollout
        with torch.no_grad():
            last_out = self.critic.forward(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
        last_value = self.value_head(last_out)  # (1,1)

        adv, ret = compute_gae(buf.rew, buf.val, buf.done, last_value, self.cfg.gamma, self.cfg.gae_lambda)

        return {
            "obs": buf.obs,
            "actions": buf.actions,
            "old_logp": buf.logp,
            "advantages": adv,
            "returns": ret,
            "val": buf.val,
        }

    def update(self, batch):
        T = batch["obs"].shape[0]
        idx = torch.randperm(T, device=self.device)
        mb = self.cfg.minibatch_size

        for _ in range(self.cfg.update_epochs):
            for s in range(0, T, mb):
                ids = idx[s:s + mb]
                mb_obs = batch["obs"][ids]
                mb_act = batch["actions"][ids]
                mb_oldp = batch["old_logp"][ids]
                mb_adv = batch["advantages"][ids]
                mb_ret = batch["returns"][ids]

                # Fresh states per minibatch
                self.actor.reset(); self.critic.reset()

                # Policy loss
                out_act = self.actor.forward(mb_obs)
                mean, log_std = policy_params_from_outstate(out_act)
                new_logp = tanh_gaussian_logp(mean, log_std, mb_act)
                ratio = (new_logp - mb_oldp).exp()
                pg1 = ratio * mb_adv
                pg2 = torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef) * mb_adv
                policy_loss = -torch.min(pg1, pg2).mean()

                # Value loss (clipped)
                out_crit = self.critic.forward(mb_obs)
                v_pred = self.value_head(out_crit)
                v_clipped = (batch["val"][ids].detach()) + (v_pred - batch["val"][ids].detach()).clamp(-self.cfg.clip_coef, self.cfg.clip_coef)
                v_loss = 0.5 * torch.max((v_pred - mb_ret) ** 2, (v_clipped - mb_ret) ** 2).mean()

                # Entropy bonus (use base Gaussian entropy as proxy)
                ent = (0.5 * (1.0 + math.log(2 * math.pi)) + log_std).mean()

                loss = policy_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * ent

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_head.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

    def train(self):
        updates = math.ceil(self.cfg.total_timesteps / self.cfg.rollout_steps)
        for u in range(updates):
            batch = self.rollout(self.cfg.rollout_steps)
            self.update(batch)
            print(f"update {u + 1}/{updates} | ret_mean={batch['returns'].mean().item():.2f} | adv_std={batch['advantages'].std().item():.3f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PPOConfig(device=device)
    agent = TwcPPOAgent(cfg)
    agent.train()

    # Evaluate the policy
    # creating a new enviroment of the problem and run some episode to see the 
    # perofmance
    env = gym.make("MountainCarContinuous-v0", render_mode="human")  # default goal_velocity=0
    obs, info = env.reset(seed=123)


    episode_over = False
    total_reward = 0
    while not episode_over:

        with torch.no_grad():
            obs_t = torch.as_tensor(obs,dtype=torch.float32,device=device).unsqueeze(0)
            out_act = agent.actor.forward(obs_t)
            mean, log_std = policy_params_from_outstate(out_act)
            a = tanh_gaussian_sample(mean, log_std)
            logp = tanh_gaussian_logp(mean, log_std, a)
            print(f"action: {a.squeeze(0).cpu().numpy()}")
            print(f"logp: {logp.squeeze(0).cpu().numpy()}")
            print(f"mean: {mean.squeeze(0).cpu().numpy()}")
            print(f"log_std: {log_std.squeeze(0).cpu().numpy()}")
            obs, reward, terminated, truncated, info = env.step(a.squeeze(0).cpu().numpy())
            total_reward += reward
            episode_over = terminated or truncated
    
    print(f"episode over, total reward: {total_reward}")

