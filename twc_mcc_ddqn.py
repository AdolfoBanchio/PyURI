# 
from utils.twc_builder import build_twc
from utils.twc_io import mcc_obs_encoder, twc_out_2_mcc_action
import gymnasium as gym
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def _global_grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach()
            total += (g * g).sum().item()
    return float(total ** 0.5)

# 
twc = build_twc(obs_encoder=mcc_obs_encoder,
                action_decoder=twc_out_2_mcc_action,
                log_stats=False)

# 
class ActorFIURI(nn.Module):
    def __init__(self, twc_module):  # your nn.Module that implements FIURI/TWC
        super().__init__()
        self.twc = twc_module

    def forward(self, obs):
        # obs: (B, obs_dim). Make sure your twc uses the training-time decoder (no @torch.no_grad)
        y = self.twc(obs)                  # (B, 2) [FWD, REV] or your chosen layout
        a = self.twc.get_action(y)      # (B, 1) torque in [-1, 1], differentiable
        return a

# 
class CriticQ(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        # optional: small init on last layer
        nn.init.uniform_(self.q[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q[-1].bias,   -3e-3, 3e-3)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q(x)  # (B, 1)


# [markdown]
# In DDPG, each learnable model (actor and critic) has two versions: a main (online) network and a target network. The main networks are the ones being actively trained — the actor learns how to select actions, and the critic learns to evaluate them. The target networks serve only to compute stable bootstrapped targets during training. To keep learning stable, the target networks are softly updated after each training step using the rule
# 
# $$ 
# \theta^{-} \leftarrow (1 - \tau)\theta^{-} + \tau \cdot \theta^{-}
# $$
# 
# where $\tau$ is a small constant (e.g., 0.005). This soft update makes the target parameters slowly track the main networks, smoothing out rapid changes and preventing divergence. During evaluation or deployment, only the main actor is used to generate actions — the target networks exist purely to stabilize the learning process.

#  [markdown]
# Off-policy data & the Replay Buffer
# 
# What it is: a big circular memory that stores past experience tuples so you can train from randomized mini-batches instead of the last on-policy rollouts only.
# 
# Why it matters: breaks temporal correlations, improves sample efficiency (you reuse data many times), and stabilizes training.
# 
# What you store (per step): (s,a,r,s,d)
# 
# Terminal handling: when computing targets later, multiply by (1−d) so you don’t bootstrap past terminal states.

# 
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=int(1e6)):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1),       dtype=np.float32)
        self.obs2= np.zeros((size, obs_dim), dtype=np.float32)
        self.done= np.zeros((size, 1),       dtype=np.float32)
        self.max_size, self.ptr, self.size = size, 0, 0

    def store(self, s, a, r, s2, d):
        self.obs[self.ptr]  = s
        self.act[self.ptr]  = a
        self.rew[self.ptr]  = r
        self.obs2[self.ptr] = s2
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs  = torch.as_tensor(self.obs[idx],  device=device),
            act  = torch.as_tensor(self.act[idx],  device=device),
            rew  = torch.as_tensor(self.rew[idx],  device=device),
            obs2 = torch.as_tensor(self.obs2[idx], device=device),
            done = torch.as_tensor(self.done[idx], device=device),
        )
        return batch

    def _idx_at(self, i: int) -> int:
        """
        Translate a logical index (0 = oldest sample) to the underlying ring-buffer slot.
        """
        if self.size == 0:
            raise IndexError("ReplayBuffer is empty")
        return (self.ptr - self.size + i) % self.max_size

    def sample_sequences(self, batch_size, seq_len, device):
        """
        Return batches of sequential transitions with shape:
          obs  : (B, seq_len, obs_dim)
          act  : (B, seq_len, act_dim)
          rew  : (B, seq_len, 1)
          obs2 : (B, seq_len, obs_dim)
          done : (B, seq_len, 1)

        Sequences are guaranteed not to cross episode boundaries: any `done`
        flag can only appear on the last element of the sequence.
        """
        if self.size < seq_len:
            raise ValueError(f"Not enough samples ({self.size}) to draw sequences of length {seq_len}")

        obs_batch = []
        act_batch = []
        rew_batch = []
        obs2_batch = []
        done_batch = []

        max_start = self.size - seq_len
        attempts = 0
        max_attempts = batch_size * 50

        while len(obs_batch) < batch_size:
            if attempts >= max_attempts:
                raise RuntimeError(
                    "Unable to sample sequential batches without crossing episode boundaries. "
                    "Consider reducing seq_len or ensuring adequate replay data."
                )
            start = np.random.randint(0, max_start + 1)
            idxs = [self._idx_at(start + offset) for offset in range(seq_len)]
            # Prevent sequences that cross terminals except possibly on last element.
            if np.any(self.done[idxs[:-1]] > 0.5):
                attempts += 1
                continue

            obs_batch.append(self.obs[idxs])
            act_batch.append(self.act[idxs])
            rew_batch.append(self.rew[idxs])
            obs2_batch.append(self.obs2[idxs])
            done_batch.append(self.done[idxs])

        obs_arr  = torch.as_tensor(np.stack(obs_batch,  axis=0), device=device)
        act_arr  = torch.as_tensor(np.stack(act_batch,  axis=0), device=device)
        rew_arr  = torch.as_tensor(np.stack(rew_batch,  axis=0), device=device)
        obs2_arr = torch.as_tensor(np.stack(obs2_batch, axis=0), device=device)
        done_arr = torch.as_tensor(np.stack(done_batch, axis=0), device=device)

        return {
            "obs": obs_arr,
            "act": act_arr,
            "rew": rew_arr,
            "obs2": obs2_arr,
            "done": done_arr,
        }

#  [markdown]
# Exploration Noise (action-space)
# 
# actor is deterministic. Without noise, you’ll stick to whatever the current policy outputs and never discover better actions.
# 
# How to apply: during data collection only, add noise to the actor’s action before stepping the env, then clip to bounds (e.g., [-1,1] torque). Disable noise during evaluation and when computing targets.
# 
# In DDQN there are two common choices Gaussian Noise or Ornstein–Uhlenbeck (OU) noise (time-correlated).

# 
def add_gaussian_noise(a, std=0.2, low=-1.0, high=1.0):
    noise = torch.randn_like(a) * std
    return torch.clamp(a + noise, low, high)


class OUNoise:
    def __init__(self, act_dim, theta=0.15, sigma=0.2, mu=0.0):
        self.theta, self.sigma = theta, sigma
        self.mu = mu
        self.state = np.zeros(act_dim, dtype=np.float32)

    def reset(self): self.state[:] = 0.0
    
    def __call__(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(*self.state.shape)
        self.state += dx
        return self.state.copy()

#  [markdown]
# Lets start with the training algorithm, first some warmup data must be collected from the enviroment.

# 
# --- Hyperparameters ---
TOTAL_STEPS       = 200_000
SAVE_EVERY        = TOTAL_STEPS // 4
UPDATE_AFTER      = 5_000
UPDATE_EVERY      = 1
NUM_UPDATE_LOOPS  = 1
EVAL_INTERVAL     = 5_000
NOISE_STD_INIT    = 0.25
NOISE_STD_FINAL   = 0.05
NOISE_DECAY_STEPS = 50_000
GAMMA, TAU        = 0.99, 0.005
BATCH_SIZE        = 128
SEQ_LEN           = 16
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ddpg_update_step(actor, critic, actor_targ, critic_targ,
                     actor_opt, critic_opt, batch, grad_clip=None):
    obs_seq  = batch['obs']   # (B, L, obs_dim)
    act_seq  = batch['act']   # (B, L, act_dim)
    rew_seq  = batch['rew']   # (B, L, 1)
    obs2_seq = batch['obs2']  # (B, L, obs_dim)
    done_seq = batch['done']  # (B, L, 1)

    seq_len = obs_seq.size(1)
    obs  = obs_seq[:, -1, :]
    act  = act_seq[:, -1, :]
    rew  = rew_seq[:, -1, :]
    obs2 = obs2_seq[:, -1, :]
    done = done_seq[:, -1, :]
    batch_size = obs_seq.size(0)

    def _snapshot_states(twc_module):
        layers = (twc_module.in_layer, twc_module.hid_layer, twc_module.out_layer)
        return [
            (
                layer.in_state.detach().clone(),
                layer.out_state.detach().clone(),
            )
            for layer in layers
        ]

    def _restore_states(twc_module, snapshot):
        layers = (twc_module.in_layer, twc_module.hid_layer, twc_module.out_layer)
        for layer, (in_state, out_state) in zip(layers, snapshot):
            layer.in_state = in_state.clone()
            layer.out_state = out_state.clone()
        twc_module.detach()

    def _reset_twc_state(twc_module, B, device, dtype):
        for layer in (twc_module.in_layer, twc_module.hid_layer, twc_module.out_layer):
            layer.reset_state(B=B, device=device, dtype=dtype)

    # ----- Critic target -----
    with torch.no_grad():
        _reset_twc_state(actor_targ.twc, batch_size, obs_seq.device, obs_seq.dtype)
        for t in range(seq_len):
            actor_targ(obs_seq[:, t, :])
        a2 = actor_targ(obs2)
        q2 = critic_targ(obs2, a2)
        y  = rew + GAMMA * (1.0 - done) * q2
        y_mean = float(y.mean().item())
        y_std  = float(y.std().item())

    # ----- Critic update -----
    critic_opt.zero_grad(set_to_none=True)
    q   = critic(obs, act)
    l_q = ((q - y) ** 2).mean()
    l_q.backward()
    critic_grad_norm = _global_grad_norm(critic.parameters())
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
    critic_opt.step()

    q_mean = float(q.mean().item())
    q_std  = float(q.std().item())

    # ----- Actor update -----
    actor_grad_norm = float('nan')
    l_pi_value = float('nan')
    actor_state_snapshot = _snapshot_states(actor.twc)
    try:
        for p in critic.parameters():
            p.requires_grad_(False)

        actor_opt.zero_grad(set_to_none=True)
        _reset_twc_state(actor.twc, batch_size, obs_seq.device, obs_seq.dtype)

        a_pi = None
        for t in range(seq_len):
            a_pi = actor(obs_seq[:, t, :])
        q_pi = critic(obs, a_pi)
        l_pi = -q_pi.mean()

        l_pi.backward()
        actor_grad_norm = _global_grad_norm(actor.parameters())
        l_pi_value = float(l_pi.item())
        actor_opt.step()
    finally:
        _restore_states(actor.twc, actor_state_snapshot)
        for p in critic.parameters():
            p.requires_grad_(True)

    actor_grad_norm = float(actor_grad_norm)

    # ----- Soft target update -----
    @torch.no_grad()
    def _soft_update(net, targ):
        for p, pt in zip(net.parameters(), targ.parameters()):
            pt.data.mul_(1 - TAU).add_(TAU * p.data)

    _soft_update(actor,  actor_targ)
    _soft_update(critic, critic_targ)

    # ===== LOGGING ADDED: return rich dict =====
    return {
        "loss/critic": float(l_q.item()),
        "loss/actor":  l_pi_value,
        "q/mean":      q_mean,
        "q/std":       q_std,
        "target/mean": y_mean,
        "target/std":  y_std,
        "grads/critic_global_norm": actor_grad_norm if False else critic_grad_norm,  # keep explicit names below
        "grads/actor_global_norm":  float(actor_grad_norm),
    }



# 
# --- Env & Networks ---
env = gym.make("MountainCarContinuous-v0")
obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
actor, critic = ActorFIURI(twc).to(DEVICE), CriticQ(obs_dim, act_dim).to(DEVICE)
actor_targ, critic_targ = deepcopy(actor), deepcopy(critic)
actor_opt = torch.optim.Adam(actor.parameters(),  lr=1e-4)
critic_opt= torch.optim.Adam(critic.parameters(), lr=1e-3)
buf = ReplayBuffer(obs_dim, act_dim, size=100_000)
writer = SummaryWriter(log_dir="runs/twc_ddpg")

# --- Helper for noise decay ---
def current_noise_std(t):
    frac = np.clip(t / NOISE_DECAY_STEPS, 0., 1.)
    return NOISE_STD_INIT + frac * (NOISE_STD_FINAL - NOISE_STD_INIT)

@torch.no_grad()
def evaluate_policy(env, actor, device, episodes=5):
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        actor.twc.reset()
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = actor(obs_t).cpu().numpy()[0]
            obs, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated
    return total / episodes

obs, _ = env.reset(seed=42)
episode_reward, episode_len = 0.0, 0
returns_window = deque(maxlen=10)

for t in tqdm(range(TOTAL_STEPS)):
    # === 1. Action selection ===
    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        a = actor(obs_t).cpu().numpy()[0]
        
    sigma = current_noise_std(t)
    noise = np.random.normal(0, current_noise_std(t), size=a.shape)
    a_env = np.clip(a + noise, -1.0, 1.0)

    # logging
    writer.add_scalar("data/noise_std", sigma, t+1)
    writer.add_scalar("data/action_noisy_mean", float(np.mean(a_env)), t+1)
    writer.add_scalar("data/action_noisy_std",  float(np.std(a_env)),  t+1)

    # === 2. Environment step ===
    next_obs, r, terminated, truncated, _ = env.step(a_env)
    done = float(terminated or truncated)
    buf.store(obs, a_env, r, next_obs, done)

    writer.add_scalar("data/buffer_fill_pct", 100.0 * buf.size / buf.max_size, t+1)

    episode_reward += r
    episode_len += 1
    obs = next_obs

    # === 3. Learn ===
    if t >= UPDATE_AFTER and t % UPDATE_EVERY == 0:
        for _ in range(NUM_UPDATE_LOOPS):
            batch = buf.sample_sequences(BATCH_SIZE, SEQ_LEN, DEVICE)
            metrics = ddpg_update_step(actor, critic,
                                       actor_targ, critic_targ,
                                       actor_opt, critic_opt, batch)
            writer.add_scalar("loss/critic", metrics["loss/critic"], t+1)
            writer.add_scalar("loss/actor",  metrics["loss/actor"],  t+1)
            writer.add_scalar("q/mean",      metrics["q/mean"],      t+1)
            writer.add_scalar("q/std",       metrics["q/std"],       t+1)
            writer.add_scalar("target/mean", metrics["target/mean"], t+1)
            writer.add_scalar("target/std",  metrics["target/std"],  t+1)
            writer.add_scalar("grads/critic_global_norm", metrics["grads/critic_global_norm"], t+1)
            writer.add_scalar("grads/actor_global_norm",  metrics["grads/actor_global_norm"],  t+1)


    # === 4. Episode end ===
    if done:
        returns_window.append(episode_reward)
        writer.add_scalar("episode/return", episode_reward, t+1)
        writer.add_scalar("episode/length", episode_len,    t+1)
        if len(returns_window) == returns_window.maxlen:
            writer.add_scalar("episode/return_avg10", float(np.mean(returns_window)), t+1)

        obs, _ = env.reset()
        actor.twc.reset()
        episode_reward, episode_len = 0.0, 0

    # === 5. Periodic evaluation ===
    if (t+1) % EVAL_INTERVAL == 0:
        eval_return = evaluate_policy(env, actor, DEVICE)
        print(f"[Eval] step={t+1}  avg_return={eval_return:.2f}")
        writer.add_scalar("eval/return_mean", eval_return, t+1)   # ===== LOGGING ADDED =====
    
    # === 6. Periodic save ===
    if t % SAVE_EVERY == 0 and t > 0:
        # save actor and critic
        torch.save(actor.state_dict(),f"twc_actor_{t}.pth")
        torch.save(critic.state_dict(),f"twc_critic_{t}.pth")

# Final model saves.
torch.save(actor.state_dict(),f"twc_actor.pth")
torch.save(critic.state_dict(),f"twc_critic.pth")

writer.flush()
writer.close()
