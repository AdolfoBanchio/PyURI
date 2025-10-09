# ppo_twc_recurrent_mcc_clean.py
import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm

from utils.twc_builder import build_twc
from utils.twc_io_wrapper import mountaincar_pair_encoder

# ---------- Tanh-squashed Gaussian ----------
class TanhNormal:
    def __init__(self, mean, log_std, eps: float = 1e-6):
        self.mean = mean
        self.log_std = log_std.clamp(-5, 2)
        self.std = torch.exp(self.log_std)
        self.base = Normal(self.mean, self.std)
        self.eps = eps
    def rsample(self):
        z = self.base.rsample()
        a = torch.tanh(z)
        return a, z
    def log_prob(self, a, z=None):
        if z is None:
            a = a.clamp(-1 + self.eps, 1 - self.eps)
            z = 0.5 * (torch.log1p(a) - torch.log1p(-a))
        lp = self.base.log_prob(z) - torch.log(1 - torch.tanh(z).pow(2) + self.eps)
        return lp.sum(-1)

# ---------- TWC state snapshot/restore ----------
def snapshot_layer(layer) -> Dict[str, torch.Tensor]:
    return {"in": layer.in_state.detach().clone(), "out": layer.out_state.detach().clone()}
def restore_layer(layer, snap: Dict[str, torch.Tensor]):
    layer.in_state.copy_(snap["in"]); layer.out_state.copy_(snap["out"]); layer.detach()
def snapshot_twc(twc) -> Dict[str, Dict[str, torch.Tensor]]:
    return {"in": snapshot_layer(twc.in_layer), "hid": snapshot_layer(twc.hid_layer), "out": snapshot_layer(twc.out_layer)}
def restore_twc(twc, snap: Dict[str, Dict[str, torch.Tensor]]):
    restore_layer(twc.in_layer, snap["in"]); restore_layer(twc.hid_layer, snap["hid"]); restore_layer(twc.out_layer, snap["out"])

# ---------- Actor-Critic with recurrent TWC ----------
class ActorCriticTWC(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.actor_twc  = build_twc(action_decoder=mountaincar_pair_encoder(), log_stats=False).to(device)
        self.critic_twc = build_twc(action_decoder=mountaincar_pair_encoder(), log_stats=False).to(device)
        with torch.no_grad():
            dummy = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
            n_out = self.actor_twc(dummy).shape[-1]
            self.actor_twc.reset(); self.critic_twc.reset()
        self.mu = nn.Linear(n_out, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.v_head = nn.Linear(n_out, 1)
        nn.init.orthogonal_(self.mu.weight, gain=0.01); nn.init.constant_(self.mu.bias, 0.0)
        nn.init.orthogonal_(self.v_head.weight, gain=1.0); nn.init.constant_(self.v_head.bias, 0.0)

    @torch.no_grad()
    def act_value_stateful(self, obs_t: torch.Tensor):
        y_a = self.actor_twc(obs_t)
        mean = self.mu(y_a)
        dist = TanhNormal(mean, self.log_std)
        a, z = dist.rsample()
        logp = dist.log_prob(a, z)
        y_v = self.critic_twc(obs_t)
        v = self.v_head(y_v).squeeze(-1)
        self.actor_twc.detach(); self.critic_twc.detach()
        return a, logp, v

    def evaluate_sequence(self, obs_seq: torch.Tensor, init_actor_state, init_critic_state):
        # obs_seq: (T, 1, obs_dim)
        restore_twc(self.actor_twc,  init_actor_state)
        restore_twc(self.critic_twc, init_critic_state)
        T = obs_seq.size(0)
        ent, vals = [], []
        for t in range(T):
            x_t = obs_seq[t]
            y_pi = self.actor_twc(x_t)
            mean = self.mu(y_pi)
            entropy = (0.5 * (1 + math.log(2 * math.pi)) + self.log_std).sum(-1)  # pre-tanh approx
            ent.append(entropy.squeeze(0))
            y_v = self.critic_twc(x_t)
            vals.append(self.v_head(y_v).squeeze(-1).squeeze(0))
            self.actor_twc.detach(); self.critic_twc.detach()
        return torch.stack(ent, 0), torch.stack(vals, 0)

    def log_prob_actions_sequence(self, obs_seq: torch.Tensor, act_seq: torch.Tensor, init_actor_state):
        restore_twc(self.actor_twc, init_actor_state)
        T = obs_seq.size(0); logps = []
        for t in range(T):
            y_pi = self.actor_twc(obs_seq[t])
            mean = self.mu(y_pi)
            dist = TanhNormal(mean, self.log_std)
            logps.append(dist.log_prob(act_seq[t].unsqueeze(0)).squeeze(0))
            self.actor_twc.detach()
        return torch.stack(logps, 0)

# ---------- Advantage ----------
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, bootstrap: float = 0.0):
    T = len(rewards)
    adv = torch.zeros(T, device=rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        next_v = bootstrap if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    ret = adv + values
    return adv, ret

# ---------- Config ----------
@dataclass
class PPOCfg:
    env_id: str = "MountainCarContinuous-v0"
    total_steps: int = 200_000
    rollout_len: int = 1024
    ppo_epochs: int = 10
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every_updates: int = 10
    use_amp: bool = True   # GPU-friendly mixed precision

# ---------- Trainer ----------
def train_ppo(cfg: PPOCfg):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    env = gym.make(cfg.env_id)
    obs, _ = env.reset(seed=cfg.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    ac = ActorCriticTWC(obs_dim, act_dim, device).to(device)
    pi_params = list(ac.actor_twc.parameters()) + list(ac.mu.parameters()) + [ac.log_std]
    vf_params = list(ac.critic_twc.parameters()) + list(ac.v_head.parameters())
    opt_pi = torch.optim.Adam(pi_params, lr=cfg.pi_lr)
    opt_vf = torch.optim.Adam(vf_params, lr=cfg.vf_lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    steps_done, updates = 0, 0
    bar = tqdm(total=cfg.total_steps, desc="PPO-TWC", ncols=100)

    while steps_done < cfg.total_steps:
        # ---- Rollout (stateful) ----
        obs_seq, act_seq, rew_seq, done_seq, val_seq, logp_old_seq = [], [], [], [], [], []
        init_actor_state = snapshot_twc(ac.actor_twc)
        init_critic_state = snapshot_twc(ac.critic_twc)

        for t in range(cfg.rollout_len):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a_t, logp_t, v_t = ac.act_value_stateful(obs_t)
            a_np = a_t.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(a_np)
            done = terminated or truncated

            obs_seq.append(obs_t.squeeze(0))
            act_seq.append(a_t.squeeze(0))
            rew_seq.append(torch.tensor(reward, dtype=torch.float32, device=device))
            done_seq.append(torch.tensor(float(done), dtype=torch.float32, device=device))
            val_seq.append(v_t.squeeze(0))
            logp_old_seq.append(logp_t.squeeze(0))

            obs = next_obs
            steps_done += 1
            if done:
                obs, _ = env.reset()
                ac.actor_twc.reset(); ac.critic_twc.reset()

            if steps_done >= cfg.total_steps:
                break

        # Bootstrap value
        with torch.no_grad():
            v_boot = ac.act_value_stateful(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)

        # Stack tensors
        obs_seq = torch.stack(obs_seq, 0)
        act_seq = torch.stack(act_seq, 0)
        rew_seq = torch.stack(rew_seq, 0)
        done_seq = torch.stack(done_seq, 0)
        val_seq = torch.stack(val_seq, 0)
        logp_old_seq = torch.stack(logp_old_seq, 0)

        # Advantages/returns
        adv_seq, ret_seq = compute_gae(rew_seq, val_seq, done_seq, cfg.gamma, cfg.lam, bootstrap=v_boot)
        adv_norm = (adv_seq - adv_seq.mean()) / (adv_seq.std() + 1e-8)

        # ---- PPO updates (recurrent replay) ----
        for _ in range(cfg.ppo_epochs):
            obs_seq_b = obs_seq.unsqueeze(1)  # (T, 1, obs)
            # Recompute values/entropy
            entropy_seq, v_seq_new = ac.evaluate_sequence(obs_seq_b, init_actor_state, init_critic_state)
            # Exact log prob of stored actions
            logp_new_seq = ac.log_prob_actions_sequence(obs_seq_b, act_seq, init_actor_state)

            ratio = torch.exp(logp_new_seq - logp_old_seq)
            surr1 = ratio * adv_norm
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_norm
            pi_loss = -(torch.min(surr1, surr2)).mean() - cfg.ent_coef * entropy_seq.mean()
            vf_loss = F.mse_loss(v_seq_new, ret_seq)

            # Policy update (AMP-friendly)
            opt_pi.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                pi_loss_amp = pi_loss  # already computed in fp32; safe to just backward
            scaler.scale(pi_loss_amp).backward()
            nn.utils.clip_grad_norm_(pi_params, cfg.max_grad_norm)
            scaler.step(opt_pi)
            scaler.update()

            # Value update
            opt_vf.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                vf_loss_amp = cfg.vf_coef * vf_loss
            scaler.scale(vf_loss_amp).backward()
            nn.utils.clip_grad_norm_(vf_params, cfg.max_grad_norm)
            scaler.step(opt_vf)
            scaler.update()

        updates += 1
        bar.update(min(cfg.rollout_len, cfg.total_steps - bar.n))  # one quiet bar for the whole training

        # Print every N updates (single concise line)
        if updates % cfg.log_every_updates == 0:
            # Avoid sync storms: move small scalars once
            pi_l = float(pi_loss.detach().cpu())
            vf_l = float(vf_loss.detach().cpu())
            ev = float(1 - (ret_seq - v_seq_new).var() / (ret_seq.var() + 1e-8))
            avg_ret = float(rew_seq.sum().detach().cpu())  # sum of last rollout rewards (single env)
            print(f"[Upd {updates:4d}] steps={steps_done:6d} | pi={pi_l:.3f} vf={vf_l:.3f} | EV={ev:.3f} | RolloutRet={avg_ret:.2f}")

    bar.close(); env.close()

# ---------- Entry ----------
if __name__ == "__main__":
    cfg = PPOCfg(
        total_steps=200_000,
        rollout_len=1024,
        ppo_epochs=10,
        ent_coef=0.00,
        log_every_updates=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=True,
    )
    train_ppo(cfg)

