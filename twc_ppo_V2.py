# ppo_twc_mcc.py
import math
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from utils.twc_builder import build_twc
from utils.twc_io_wrapper import mountaincar_pair_encoder

# ---------- Helpers: tanh-squashed normal with log-prob correction ----------
class TanhNormal:
    def __init__(self, mean, log_std, eps=1e-6):
        self.mean = mean
        self.log_std = log_std.clamp(-5, 2)   # clamp for stability
        self.std = torch.exp(self.log_std)
        self.base = Normal(self.mean, self.std)
        self.eps = eps

    def sample(self):
        z = self.base.rsample()               # reparam trick
        a = torch.tanh(z)
        return a, z

    def log_prob(self, a, z=None):
        # if z not provided, invert tanh: atanh(a)
        if z is None:
            a = a.clamp(-1 + self.eps, 1 - self.eps)
            z = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh
        log_prob = self.base.log_prob(z) - torch.log(1 - torch.tanh(z).pow(2) + self.eps)
        return log_prob.sum(-1)

# ---------- Rollout buffer (on-policy) ----------
class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim, device):
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False
        self.obs  = torch.zeros(size, obs_dim, device=device)
        self.act  = torch.zeros(size, act_dim, device=device)
        self.rew  = torch.zeros(size, device=device)
        self.done = torch.zeros(size, device=device)
        self.val  = torch.zeros(size, device=device)
        self.logp = torch.zeros(size, device=device)
        self.adv  = torch.zeros(size, device=device)
        self.ret  = torch.zeros(size, device=device)

    def add(self, obs, act, rew, done, val, logp):
        self.obs[self.ptr]  = obs
        self.act[self.ptr]  = act
        self.rew[self.ptr]  = rew
        self.done[self.ptr] = done
        self.val[self.ptr]  = val
        self.logp[self.ptr] = logp
        self.ptr += 1
        if self.ptr >= self.size:
            self.full = True
            self.ptr = 0

    def compute_gae(self, last_val, gamma=0.99, lam=0.95):
        T = self.size
        adv = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - self.done[t]
            next_val = last_val if t == T - 1 else self.val[t + 1]
            delta = self.rew[t] + gamma * next_val * mask - self.val[t]
            gae = delta + gamma * lam * mask * gae
            adv[t] = gae
        ret = adv + self.val
        # normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.adv[:] = adv
        self.ret[:] = ret

    def get_minibatches(self, batch_size, shuffle=True):
        idxs = torch.arange(self.size, device=self.device)
        if shuffle:
            idxs = idxs[torch.randperm(self.size, device=self.device)]
        for start in range(0, self.size, batch_size):
            mb_idx = idxs[start:start + batch_size]
            yield (self.obs[mb_idx],
                   self.act[mb_idx],
                   self.adv[mb_idx],
                   self.ret[mb_idx],
                   self.logp[mb_idx],
                   self.val[mb_idx])

# ---------- Actor-Critic with TWC torso ----------
class ActorCriticTWC(nn.Module):
    def __init__(self, obs_dim, act_dim, device):
        super().__init__()
        self.device = device
        # TWC torso for actor and critic (separate instances to avoid value/policy interference)
        self.actor_twc  = build_twc(action_decoder=mountaincar_pair_encoder(), use_json_w=True).to(device)
        self.critic_twc = build_twc(action_decoder=mountaincar_pair_encoder(), use_json_w=True).to(device)

        # The TWC forward returns a vector (n_out). Map it to action-mean (act_dim) and value (scalar).
        # We don’t assume n_out; we infer after a dummy pass.
        with torch.no_grad():
            dummy = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
            y_a = self.actor_twc(dummy)   # (1, n_out)
            n_out = y_a.shape[-1]
            # reset states afterwards
            self.actor_twc.reset()
            self.critic_twc.reset()

        self.policy_mean = nn.Linear(n_out, act_dim)
        self.log_std     = nn.Parameter(torch.full((act_dim,), -0.5))  # learnable global log_std

        self.value_head  = nn.Linear(n_out, 1)

        # Optional: small init for heads
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.constant_(self.policy_mean.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    @torch.no_grad()
    def reset_states(self):
        self.actor_twc.reset()
        self.critic_twc.reset()

    def detach_states(self):
        self.actor_twc.detach()
        self.critic_twc.detach()

    def act(self, obs_tensor):
        # obs_tensor: (B, obs_dim)
        y = self.actor_twc(obs_tensor)        # (B, n_out)
        mean = self.policy_mean(y)            # (B, act_dim)
        dist = TanhNormal(mean, self.log_std)
        a, z = dist.sample()                  # tanh-squashed
        logp = dist.log_prob(a, z)            # (B,)
        # critic value (no grad-sharing)
        with torch.no_grad():
            v = self.value(obs_tensor)
        return a, logp, v

    def evaluate_actions(self, obs_tensor, actions):
        # Used during PPO updates (no env stepping), requires gradients
        y_pi = self.actor_twc(obs_tensor)
        mean = self.policy_mean(y_pi)
        dist = TanhNormal(mean, self.log_std)
        # actions are already in [-1,1], compute logp with inverse
        logp = dist.log_prob(actions)
        entropy = (0.5 * (1 + math.log(2 * math.pi)) + self.log_std).sum(-1)  # base Normal entropy; not exact post-tanh
        v = self.value(obs_tensor)
        return logp, entropy, v

    def value(self, obs_tensor):
        y_v = self.critic_twc(obs_tensor)
        return self.value_head(y_v).squeeze(-1)  # (B,)

# ---------- PPO Trainer ----------
class PPOAgent:
    def __init__(
        self,
        env_id="MountainCarContinuous-v0",
        total_steps=200_000,
        rollout_len=2048,
        epochs=10,
        minibatch_size=256,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=1e-3,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=1.0,
        seed=123,
        render=False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(env_id, render_mode="human" if render else None)
        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.act_dim = int(np.prod(self.env.action_space.shape))
        self.ac = ActorCriticTWC(self.obs_dim, self.act_dim, self.device).to(self.device)

        # Two optimizers (separate LR often helps)
        policy_params = list(self.ac.actor_twc.parameters()) + \
                        list(self.ac.policy_mean.parameters()) + \
                        [self.ac.log_std]
        value_params  = list(self.ac.critic_twc.parameters()) + \
                        list(self.ac.value_head.parameters())
        self.opt_pi = torch.optim.Adam(policy_params, lr=pi_lr)
        self.opt_vf = torch.optim.Adam(value_params,  lr=vf_lr)

        self.total_steps = total_steps
        self.rollout_len = rollout_len
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.rng = np.random.default_rng(seed)
        self.env.reset(seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self):
        obs, _ = self.env.reset()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.ac.reset_states()

        buf = RolloutBuffer(self.rollout_len, self.obs_dim, self.act_dim, self.device)
        steps_done = 0
        ep_return, ep_len = 0.0, 0

        while steps_done < self.total_steps:
            # ------- Rollout (collect on-policy data) -------
            buf.ptr = 0; buf.full = False
            for t in range(self.rollout_len):
                with torch.no_grad():
                    a_t, logp_t, v_t = self.ac.act(obs_t)   # (1, act_dim), (1,), (1,)
                a_np = a_t.squeeze(0).cpu().numpy()
                # step env
                next_obs, rew, term, trunc, _ = self.env.step(a_np)
                done = term or trunc

                # store
                buf.add(
                    obs_t.squeeze(0),
                    a_t.squeeze(0),
                    torch.tensor(rew, dtype=torch.float32, device=self.device),
                    torch.tensor(float(done), dtype=torch.float32, device=self.device),
                    v_t.squeeze(0),
                    logp_t.squeeze(0),
                )

                ep_return += rew; ep_len += 1
                steps_done += 1

                # Prepare next obs
                obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                # IMPORTANT: keep neuron states but detach graph between env steps
                self.ac.detach_states()

                if done:
                    obs, _ = self.env.reset()
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    self.ac.reset_states()
                    print(f"[Episode] return={ep_return:.2f} len={ep_len}")
                    ep_return, ep_len = 0.0, 0

            # bootstrap for GAE
            with torch.no_grad():
                last_val = self.ac.value(obs_t).squeeze(0)
            buf.compute_gae(last_val, gamma=self.gamma, lam=self.lam)

            # ------- PPO Updates -------
            for _ in range(self.epochs):
                for obs_b, act_b, adv_b, ret_b, logp_old_b, _ in buf.get_minibatches(self.minibatch_size):
                    self.ac.detach_states()
		    # policy update
                    self.opt_pi.zero_grad(set_to_none=True)
                    logp_b, entropy_b, v_pred_dummy = self.ac.evaluate_actions(obs_b, act_b)
                    ratio = torch.exp(logp_b - logp_old_b)
                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_b
                    pi_loss = -(torch.min(surr1, surr2)).mean() - self.ent_coef * entropy_b.mean()
                    pi_loss.backward()
                    nn.utils.clip_grad_norm_(self.opt_pi.param_groups[0]['params'], self.max_grad_norm)
                    self.opt_pi.step()

                    # value update
                    self.opt_vf.zero_grad(set_to_none=True)
                    v_pred = self.ac.value(obs_b)
                    vf_loss = F.mse_loss(v_pred, ret_b)
                    (self.vf_coef * vf_loss).backward()
                    nn.utils.clip_grad_norm_(self.opt_vf.param_groups[0]['params'], self.max_grad_norm)
                    self.opt_vf.step()

            # (optional) simple logging
            with torch.no_grad():
                avg_v = buf.val.mean().item()
                avg_adv = buf.adv.mean().item()
                print(f"[Update] steps={steps_done} pi_loss={pi_loss.item():.4f} vf_loss={vf_loss.item():.4f} "
                      f"V={avg_v:.3f} Adv={avg_adv:.3f}")

        self.env.close()

# ---------- Entry ----------
if __name__ == "__main__":
    agent = PPOAgent(
        env_id="MountainCarContinuous-v0",
        total_steps=200_000,
        rollout_len=2048,
        epochs=10,
        minibatch_size=256,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=1e-3,
        ent_coef=0.00,   # can try 0.01
        vf_coef=0.5,
        render=False,
        seed=123,
    )
    agent.train()
