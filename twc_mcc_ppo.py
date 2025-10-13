import math
from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import trange

from utils.twc_builder import build_twc
from utils.twc_io import mcc_obs_encoder, twc_out_2_mcc_drive


class Rollout:
    """
    Rollout buffer class to store observations, actions, etc...
    used in the rollout stage of each episode.
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
        self.obs[i].copy_(obs.view(-1))
        self.actions[i].copy_(action.view(-1))
        self.logp[i].copy_(logp.view(-1))
        self.rew[i, 0] = reward
        self.done[i, 0] = float(done)
        self.val[i].copy_(value.view(-1))
        self.ptr += 1

    def reset(self):
        self.ptr = 0


@torch.no_grad()
def compute_gae(
    rew: torch.Tensor,
    val: torch.Tensor,
    done: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        nonterm = 1.0 - done[t:t + 1]  # (1,1)
        delta = rew[t:t + 1] + gamma * next_value * next_nonterm - val[t:t + 1]
        lastgaelam = delta + gamma * lam * next_nonterm * lastgaelam
        adv[t:t + 1] = lastgaelam
        next_value = val[t:t + 1]
        next_nonterm = nonterm

    ret = adv + val
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


@dataclass
class PPOConfig:
    env_id: str = "MountainCarContinuous-v0"
    total_timesteps: int = 50_000
    save_every: int = 10_000
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
    device: str = "cpu"


def _atanh(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    x = x.clamp(-1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def tanh_normal_sample(mean: torch.Tensor, log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_std = torch.clamp(log_std, -5.0, 2.0)
    std = torch.exp(log_std)
    normal = Normal(mean, std)
    pre_tanh = normal.rsample()
    action = torch.tanh(pre_tanh)
    log_prob = normal.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
    log_prob = log_prob.sum(-1, keepdim=True)
    entropy = normal.entropy().sum(-1, keepdim=True)
    return action, log_prob, entropy


def tanh_normal_log_prob(action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    log_std = torch.clamp(log_std, -5.0, 2.0)
    std = torch.exp(log_std)
    action = action.clamp(-0.999999, 0.999999)
    pre_tanh = _atanh(action)
    normal = Normal(mean, std)
    log_prob = normal.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
    return log_prob.sum(-1, keepdim=True)


def normal_entropy(log_std: torch.Tensor) -> torch.Tensor:
    return (0.5 * (1.0 + math.log(2 * math.pi)) + log_std).sum(dim=-1, keepdim=True)


class PPOActor(nn.Module):
    def __init__(self, twc_module: nn.Module, action_dim: int, init_log_std: float = -0.5):
        super().__init__()
        self.twc = twc_module
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.twc(obs)
        drive = self.twc.get_action(y)  # (B, action_dim)
        log_std = self.log_std.view(1, -1).expand_as(drive)
        return drive, log_std

    def reset_state(self):
        self.twc.reset()

    def detach_state(self):
        self.twc.detach()


class ValueCritic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def _to_tensor(x, device):
    return torch.as_tensor(x, device=device, dtype=torch.float32).unsqueeze(0)


def _ppo_update(
    actor: PPOActor,
    critic: ValueCritic,
    actor_opt: optim.Optimizer,
    critic_opt: optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    config: PPOConfig,
) -> Dict[str, float]:
    obs = batch["obs"]
    actions = batch["actions"]
    logp_old = batch["logp"]
    adv = batch["adv"]
    returns = batch["ret"]

    T = obs.size(0)
    adv = adv.detach()
    logp_old = logp_old.detach()

    metrics = {"actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clipfrac": 0.0, "kl": 0.0}
    minibatch_size = min(config.minibatch_size, T)

    for _ in range(config.update_epochs):
        perm = torch.randperm(T, device=obs.device)

        actor_epoch_loss = 0.0
        value_epoch_loss = 0.0
        entropy_epoch = 0.0
        clipfrac_epoch = 0.0
        kl_epoch = 0.0
        num_minibatches = 0

        for start in range(0, T, minibatch_size):
            end = start + minibatch_size
            mb_idx = perm[start:end]
            mb_obs = obs[mb_idx]
            mb_actions = actions[mb_idx]
            mb_logp_old = logp_old[mb_idx]
            mb_adv = adv[mb_idx]
            mb_returns = returns[mb_idx]

            mb_adv = mb_adv.detach()
            mb_logp_old = mb_logp_old.detach()
            mb_returns = mb_returns.detach()

            actor.reset_state()
            mean, log_std = actor(mb_obs)
            logp = tanh_normal_log_prob(mb_actions, mean, log_std)
            ratio = torch.exp(logp - mb_logp_old)
            clipped_ratio = torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)

            surr1 = ratio * mb_adv
            surr2 = clipped_ratio * mb_adv
            actor_loss = -torch.min(surr1, surr2).mean()

            entropy = normal_entropy(log_std).mean()
            value = critic(mb_obs)
            value_loss = 0.5 * (value - mb_returns).pow(2).mean()

            approx_kl = (mb_logp_old - logp).mean().abs()
            clipfrac = (ratio.gt(1.0 + config.clip_coef) | ratio.lt(1.0 - config.clip_coef)).float().mean()

            actor_opt.zero_grad()
            (actor_loss - config.ent_coef * entropy).backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            actor_opt.step()

            critic_opt.zero_grad()
            (config.vf_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            critic_opt.step()

            actor_epoch_loss += actor_loss.item()
            value_epoch_loss += value_loss.item()
            entropy_epoch += entropy.item()
            clipfrac_epoch += clipfrac.item()
            kl_epoch += approx_kl.item()
            num_minibatches += 1

        if num_minibatches > 0:
            metrics = {
                "actor_loss": actor_epoch_loss / num_minibatches,
                "value_loss": value_epoch_loss / num_minibatches,
                "entropy": entropy_epoch / num_minibatches,
                "clipfrac": clipfrac_epoch / num_minibatches,
                "kl": kl_epoch / num_minibatches,
            }

    return metrics


def train(config: PPOConfig):
    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    env = gym.make(config.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=100)
    env.action_space.seed(config.seed)

    obs, _ = env.reset(seed=config.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    twc_module = build_twc(
        obs_encoder=mcc_obs_encoder,
        action_decoder=twc_out_2_mcc_drive,
        log_stats=False,
    ).to(device)

    actor = PPOActor(twc_module, act_dim).to(device)
    critic = ValueCritic(obs_dim).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=config.lr)
    critic_opt = optim.Adam(critic.parameters(), lr=config.lr)
    rollout = Rollout(config.rollout_steps, obs_dim, device)

    global_step = 0
    num_updates = math.ceil(config.total_timesteps / config.rollout_steps)

    obs_tensor = _to_tensor(obs, device)
    actor.reset_state()

    pbar = trange(num_updates, desc="PPO", leave=False)
    episode_returns = []
    episode_lengths = []

    for update in pbar:
        if global_step >= config.total_timesteps:
            break

        rollout.reset()
        steps_this_iter = min(config.rollout_steps, config.total_timesteps - global_step)
        last_done = torch.zeros(1, 1, device=device)

        for _ in range(steps_this_iter):
            with torch.no_grad():
                mean, log_std = actor(obs_tensor)
                action, log_prob, _ = tanh_normal_sample(mean, log_std)
                value = critic(obs_tensor)

            env_action = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

            rollout.add(obs_tensor, action, log_prob, reward, done, value)
            global_step += 1
            last_done.fill_(float(done))

            if "episode" in info:
                episode_returns.append(info["episode"]["r"])
                episode_lengths.append(info["episode"]["l"])

            if done:
                obs, _ = env.reset()
                actor.reset_state()
            else:
                obs = next_obs

            obs_tensor = _to_tensor(obs, device)
            actor.detach_state()

            if global_step >= config.total_timesteps:
                break

        with torch.no_grad():
            if last_done.item() > 0.5:
                last_value = torch.zeros(1, 1, device=device)
            else:
                last_value = critic(obs_tensor)

        T = rollout.ptr
        adv, ret = compute_gae(
            rollout.rew[:T],
            rollout.val[:T],
            rollout.done[:T],
            last_value,
            config.gamma,
            config.gae_lambda,
        )

        batch = {
            "obs": rollout.obs[:T],
            "actions": rollout.actions[:T],
            "logp": rollout.logp[:T],
            "adv": adv[:T],
            "ret": ret[:T],
            "done": rollout.done[:T],
        }

        metrics = _ppo_update(actor, critic, actor_opt, critic_opt, batch, config)

        mean_return = float(torch.tensor(episode_returns[-10:], dtype=torch.float32).mean()) if episode_returns else 0.0
        mean_ep_len = float(torch.tensor(episode_lengths[-10:], dtype=torch.float32).mean()) if episode_lengths else 0.0
        pbar.set_postfix(
            steps=global_step,
            return_mean=f"{mean_return:.1f}",
            len_mean=f"{mean_ep_len:.1f}",
            loss_pi=f"{metrics['actor_loss']:.3f}",
            loss_v=f"{metrics['value_loss']:.3f}",
        )

        if global_step % cfg.save_every == 0:
            torch.save(actor.state_dict(),f"models/twc_ppo_actor_{global_step}.pth")


    env.close()
    torch.save(actor.state_dict(),f"models/twc_ppo_actor_final.pth")
    return {
        "returns": episode_returns,
        "lengths": episode_lengths,
    }


if __name__ == "__main__":
    cfg = PPOConfig()
    train(cfg)

