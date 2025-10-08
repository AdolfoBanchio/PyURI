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

def action_log_prob_from_outstate(
    out_state: torch.Tensor,
    action: torch.Tensor,
    *,
    gain: float = 1.0,
    min_log_std: float = -5.0,
    max_log_std: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    assert out_state.ndim == 2 and out_state.shape[1] == 2 and action.shape[-1] == 1
    rev = out_state[:, 0]
    fwd = out_state[:, 1]
    base_mean = gain * (fwd - rev)
    raw = fwd + rev
    log_std = min_log_std + 0.5 * (torch.tanh(raw) + 1.0) * (max_log_std - min_log_std)
    log_std = torch.clamp(log_std, min_log_std, max_log_std)
    std = torch.exp(log_std)
    var = std * std

    a = action.squeeze(-1)
    u = atanh_safe(a, eps=eps)                 # invert tanh
    LOG_2PI = 1.8378770664093453
    logp_base = -0.5 * (((u - base_mean) ** 2) / (var + eps) + 2.0 * log_std + LOG_2PI)
    squash = torch.log(1.0 - a * a + eps)
    return (logp_base - squash).unsqueeze(-1)  # (B,1)

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
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    seed: int = 0
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ValueHead(nn.Module):
    def __init__(self, twc: nn.Module,in_dim: int = 2):
        super().__init__()
        self.twc = twc
        self.v = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.v(x)


class TwcPPOAgent:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.env = gym.make(cfg.env_id)
        try:
            self.env.reset(seed=cfg.seed)
            self.env.action_space.seed(cfg.seed)
            self.env.observation_space.seed(cfg.seed)
        except Exception:
            pass

        self.obs_space = self.env.observation_space
        self.act_space = self.env.action_space

        net = build_twc()
        self.wrapper = TwcIOWrapper(
            net=net,
            device=cfg.device,
            obs_encoder=mountaincar_pair_encoder(),   
            action_decoder=stochastic_action_decoder  # used only during rollout
        )

        self.value_head = ValueHead(twc=build_twc(),in_dim=2).to(cfg.device)

        params = list(self.wrapper.net.parameters()) + list(self.value_head.parameters())
        self.optimizer = optim.Adam(params, lr=cfg.lr)

        self.device = torch.device(cfg.device)
    
    @torch.no_grad()
    def rollout(self, steps: int):
        buf = Rollout(steps, obs_dim=self.obs_space.shape[0], device=self.device)

        obs, _ = self.env.reset()
        self.wrapper.reset()
        
        for t in range(steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,2)

            # forward actor: out_state -> decode to (action, logp, ...)
            out_state = self.wrapper.step(obs_t)             # (1,2)
            pi = stochastic_action_decoder(out_state)        # PolicyOut
            a, logp = pi.action, pi.log_prob                 # (1,1), (1,1)

            # critic value (shared TWC: reuse same out_state)
            v = self.value_head(out_state)                   # (1,1)
            
            # env step
            act_np = a.squeeze(0).detach().cpu().numpy()
            next_obs, reward, terminated, truncated, _ = self.env.step(act_np)
            done = terminated or truncated

            # store (detached)
            buf.add(obs_t.detach(), a.squeeze(0).detach(), logp.squeeze(0).detach(), reward, done, v.squeeze(0).detach())

            obs = next_obs
            if done:
                obs, _ = self.env.reset()
                self.wrapper.reset()

        # bootstrap
        last_out = self.wrapper.step(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
        last_value = self.value_head(last_out)  # (1,1)

        adv, ret = compute_gae(buf.rew, buf.val, buf.done, last_value, self.cfg.gamma, self.cfg.gae_lambda)

        batch = {
            "obs": buf.obs, 
            "actions": buf.actions, 
            "old_logp": buf.logp,
            "advantages": adv, 
            "returns": ret
        }

        return batch

    
    def update(self, batch):
        T = batch["obs"].shape[0]
        idx = torch.randperm(T, device=self.device)
        mb = self.cfg.minibatch_size

        for _ in range(self.cfg.update_epochs):
            for s in range(0, T, mb):
                ids = idx[s:s+mb]
                mb_obs  = batch["obs"][ids]
                mb_act  = batch["actions"][ids]
                mb_oldp = batch["old_logp"][ids]
                mb_adv  = batch["advantages"][ids]
                mb_ret  = batch["returns"][ids]

                # recompute with current policy/value
                # (Important: one fresh forward; single backward later)
                self.wrapper.reset()
                # self.wrapper.detach()

                out_state = self.wrapper.step(mb_obs)                # (B,2)
                new_logp  = action_log_prob_from_outstate(out_state, mb_act)  # (B,1)
                v         = self.value_head(out_state)               # (B,1)

                ratio = (new_logp - mb_oldp).exp()
                pg_loss = torch.max(-mb_adv * ratio,
                                    -mb_adv * torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef)
                                   ).mean()
                v_loss  = 0.5 * (v - mb_ret).pow(2).mean()
                loss = pg_loss + self.cfg.vf_coef * v_loss  # + self.cfg.ent_coef * entropy.mean() if you log entropy

                # --- DIAGNOSTICS ---
                with torch.no_grad():
                    approx_kl = (mb_oldp - new_logp).mean().abs().item()
                    clip_frac = ( (ratio < 1 - self.cfg.clip_coef) | (ratio > 1 + self.cfg.clip_coef) ).float().mean().item()
                    # check any actor param has nonzero grad after backward
                # -------------------
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.wrapper.net.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_head.parameters(), self.cfg.max_grad_norm)
                
                actor_params  = list(self.wrapper.net.parameters())
                critic_params = list(self.value_head.parameters())
                gn_actor  = torch.nn.utils.clip_grad_norm_(actor_params,  self.cfg.max_grad_norm).item()
                gn_critic = torch.nn.utils.clip_grad_norm_(critic_params, self.cfg.max_grad_norm).item()

                
                self.optimizer.step()
                print(f"pg={pg_loss.item():.3f} v={v_loss.item():.3f} kl≈{approx_kl:.4f} clip={clip_frac:.2f} "
                         f"| ||grad||_actor={gn_actor:.3e} ||grad||_critic={gn_critic:.3e}")

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
