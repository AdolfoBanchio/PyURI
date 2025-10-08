import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.twc_builder import build_twc
from utils.twc_io_wrapper import TwcIOWrapper, mountaincar_pair_encoder, stochastic_action_decoder

# ============= HELPER FUNCTIONS/CLASSES ===============
def atanh_safe(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # clamp to avoid ±1
    a = a.clamp(-1.0 + eps, 1.0 - eps)
    # atanh(a) = 0.5 * (log1p(a) - log1p(-a))
    return 0.5 * (torch.log1p(a) - torch.log1p(-a))

LOG_2PI = 1.8378770664093453

def policy_params_from_outstate(out_state, *, gain=1.0, min_log_std=-5.0, max_log_std=2.0):
    # out_state: (B,2) -> [rev, fwd]
    rev, fwd = out_state[:, 0], out_state[:, 1]
    mean = gain * (fwd - rev)
    raw  = fwd + rev
    log_std = min_log_std + 0.5 * (torch.tanh(raw) + 1.0) * (max_log_std - min_log_std)
    log_std = torch.clamp(log_std, min_log_std, max_log_std)
    return mean.unsqueeze(-1), log_std.unsqueeze(-1)  # (B,1), (B,1)

def tanh_gaussian_logp(mean, log_std, action, eps=1e-6):
    # action in [-1,1], shape (B,1)
    std = log_std.exp()
    u   = atanh_safe(action, eps=eps)                       # pre-squash
    logp_base = -0.5 * (((u - mean)**2)/(std*std + eps) + 2.0*log_std + LOG_2PI)
    squash = torch.log(1.0 - action*action + eps)
    return logp_base - squash  # (B,1)

@torch.no_grad()
def tanh_gaussian_sample(mean, log_std):
    std = log_std.exp()
    u   = mean + std * torch.randn_like(std)
    a   = torch.tanh(u)
    return a

class Rollout:
    """ 
        Rollout buffer class to store observations, actions, etc...
        used in the rollout stage of each episode. 
    """
    def __init__(self, T: int, obs_dim: int, device: torch.device):
        self.obs     = torch.zeros(T, obs_dim, device=device)
        self.actions = torch.zeros(T, 1, device=device)
        self.logp    = torch.zeros(T, 1, device=device)
        self.rew     = torch.zeros(T, 1, device=device)
        self.done    = torch.zeros(T, 1, device=device)
        self.val     = torch.zeros(T, 1, device=device)
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
    Shapes: rew/val/done: (T,1), last_value: (1,1)
    Uses dones[t] and bootstraps from last_value if rollout ends mid-episode.
    """
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    next_value = last_value
    next_nonterm = torch.ones(1, 1, device=rew.device)
    lastgaelam = torch.zeros(1, 1, device=rew.device)

    for t in reversed(range(T)):
        nonterm = 1.0 - done[t:t+1]  # (1,1)
        delta = rew[t:t+1] + gamma * next_value * next_nonterm - val[t:t+1]
        lastgaelam = delta + gamma * lam * next_nonterm * lastgaelam
        adv[t:t+1] = lastgaelam
        next_value = val[t:t+1]
        next_nonterm = nonterm

    ret = adv + val
    # normalize advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


# ============== PPO AGENT ============ 
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
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ValueHead(nn.Module):
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
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.device = cfg.device
        self.env = gym.make(cfg.env_id, render_mode="rgb_array")
        try:
            self.env.reset(seed=cfg.seed)
            self.env.action_space.seed(cfg.seed)
            self.env.observation_space.seed(cfg.seed)
        except Exception:
            pass

        self.obs_space = self.env.observation_space
        self.act_space = self.env.action_space

        net_actor = build_twc()
        net_critic = build_twc()
        
        self.actor = TwcIOWrapper(
            net=net_actor,
            device=cfg.device,
            obs_encoder=mountaincar_pair_encoder(),
            action_decoder=None,  # we’ll sample ourselves via tanh_gaussian_*
        )
        
        self.critic = TwcIOWrapper(
            net=net_critic,
            device=cfg.device,
            obs_encoder=mountaincar_pair_encoder(),
            action_decoder=None,
        )
        
        self.value_head = ValueHead(in_dim=2).to(cfg.device)
        
        # separate (or shared) optimizer — start shared for simplicity
        params = list(self.actor.net.parameters()) + list(self.critic.net.parameters()) + list(self.value_head.parameters())
        self.optimizer = optim.Adam(params, lr=cfg.lr)
        assert any(p.requires_grad for p in self.actor.net.parameters())
        self.ent_coef     = 0.01     # small exploration
        self.target_kl    = 0.03     # early-stop threshold per epoch
        self.vf_clip_coef = 0.2      # value loss clipping
        
    def rollout(self, steps: int):
        buf = Rollout(steps, obs_dim=self.obs_space.shape[0], device=self.device)

        obs, _ = self.env.reset()
        self.actor.reset(); self.critic.reset()

        for t in range(steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            # actor forward
            with torch.no_grad():
                out_act = self.actor.step(obs_t)             # (1,2)
            mean, log_std = policy_params_from_outstate(out_act)
            a = tanh_gaussian_sample(mean, log_std)      # (1,1)
            logp = tanh_gaussian_logp(mean, log_std, a)  # (1,1)

            # critic forward
            with torch.no_grad():
                out_crit = self.critic.step(obs_t)           # (1,2)
            v = self.value_head(out_crit)                # (1,1)

            act_np = a.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = self.env.step(act_np)
            done = terminated or truncated

            buf.add(obs_t, a.squeeze(0), logp.squeeze(0), reward, done, v.squeeze(0))
            obs = next_obs

            if done:
                obs, _ = self.env.reset()
                self.actor.reset(); self.critic.reset()

        # bootstrap from critic
        with torch.no_grad():
            last_out = self.critic.step(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
        last_value = self.value_head(last_out)  # (1,1)

        adv, ret = compute_gae(buf.rew, buf.val, buf.done, last_value, self.cfg.gamma, self.cfg.gae_lambda)

        return {
            "obs": buf.obs,
            "actions": buf.actions,
            "old_logp": buf.logp,
            "advantages": adv,
            "returns": ret,
            "val": buf.val
        }

    def update(self, batch):
        T = batch["obs"].shape[0]
        idx = torch.randperm(T, device=self.device)
        mb = self.cfg.minibatch_size

        for epoch in range(self.cfg.update_epochs):
            approx_kl_epoch = []
            for s in range(0, T, mb):
                ids = idx[s:s+mb]
                mb_obs  = batch["obs"][ids]
                mb_act  = batch["actions"][ids]
                mb_oldp = batch["old_logp"][ids]
                mb_adv  = batch["advantages"][ids]
                mb_ret  = batch["returns"][ids]

                # fresh states per minibatch
                self.actor.detach()
                self.critic.detach()

                out_act = self.actor.step(mb_obs)                     # (B,2)
                mean, log_std = policy_params_from_outstate(out_act)
                new_logp = tanh_gaussian_logp(mean, log_std, mb_act)  # (B,1)
                ratio = (new_logp - mb_oldp).exp()

                pg1 =  ratio * mb_adv
                pg2 =  torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef) * mb_adv
                policy_loss = -torch.min(pg1, pg2).mean()

                # critic
                out_crit = self.critic.step(mb_obs)
                v_pred = self.value_head(out_crit)
                # clipped value loss like PPO2
                v_clipped = (batch["val"][ids].detach() if "val" in batch else v_pred.detach()) + (v_pred - (batch["val"][ids].detach() if "val" in batch else v_pred.detach())).clamp(-self.vf_clip_coef, self.vf_clip_coef)
                v_loss = 0.5 * torch.max((v_pred - mb_ret)**2, (v_clipped - mb_ret)**2).mean()

                # entropy bonus (use pre-squash Gaussian entropy as proxy)
                ent = (0.5 * (1.0 + math.log(2*math.pi)) + log_std).mean()

                loss = policy_loss + self.cfg.vf_coef * v_loss - self.ent_coef * ent

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                actor_grads = [p.grad for p in self.actor.net.parameters() if p.grad is not None]
                assert len(actor_grads) > 0, "no actor gradients — check no_grad/detach"
                nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.net.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_head.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

            with torch.no_grad():
                approx_kl = (mb_oldp - new_logp).mean().abs().item()
                clipfrac  = ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()
                print(f"pg={policy_loss.item():.3f} v={v_loss.item():.3f} kl≈{approx_kl:.4f} clip={clipfrac:.2f}")
            # KL early stop per epoch
            if np.mean(approx_kl_epoch) > 1.5 * self.target_kl:
                print(f"Early stop: KL {np.mean(approx_kl_epoch):.4f} > {1.5*self.target_kl:.4f}")
                break

            for name, param in self.actor.net.named_parameters():
                if param.grad is not None:
                    print(f"Gradient for {name}: {param.grad}")
                else:
                    print(f"No gradient for {name}")


    def train(self):
        # derive number of updates from total_timesteps and rollout_steps
        updates = math.ceil(self.cfg.total_timesteps / self.cfg.rollout_steps)
        obs, _ = self.env.reset()
        for u in range(updates):
            batch = self.rollout(self.cfg.rollout_steps)
            self.update(batch)
            # quick debug scalar
            print(f"update {u+1}/{updates} | "
                  f"ret_mean={batch['returns'].mean().item():.2f} | "
                  f"adv_std={batch['advantages'].std().item():.3f}")


if __name__ == "__main__":
    device = (
        torch.device(0)
        if torch.cuda.is_available() 
        else torch.device("cpu")
    )
    cfg = PPOConfig(device=device)
    agent = TwcPPOAgent(cfg)
    agent.train()
