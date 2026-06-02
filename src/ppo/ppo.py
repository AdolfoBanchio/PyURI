import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import optuna

from copy import deepcopy
from dataclasses import dataclass, asdict, field
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fiuri import PyUriTwc, model
from mlp import ValueCriticInvPen
from utils import EpisodeRolloutBuffer, RolloutBatch


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    """Hyperparameters for PPO + BPTT with a TWC actor."""

    # --- Training loop ---
    max_train_steps:       int   = 300_000
    episodes_per_update:   int   = 20        # rollout length in episodes
    ppo_epochs:            int   = 10        # gradient epochs per update
    mini_batch_size:       int   = 256       # flat steps per mini-batch
    device:                str   = "cuda" if torch.cuda.is_available() else "cpu"
    seed:                  int   = 42

    # --- Evaluation ---
    eval_interval_updates: int   = 5         # eval every N update rounds
    eval_episodes:         int   = 50

    # --- BPTT ---
    burn_in_length:        int   = 8         # steps discarded from BPTT loss

    # --- PPO hyperparameters ---
    actor_lr:              float = 3e-4
    critic_lr:             float = 1e-3
    gamma:                 float = 0.99
    lam:                   float = 0.95      # GAE lambda
    clip_eps:              float = 0.2       # PPO clip ratio
    entropy_coef:          float = 0.01
    value_loss_coef:       float = 0.5
    max_grad_norm:         float = 0.5

    # --- Initial log_std ---
    log_std_init:          float = -0.5      # ~ std ≈ 0.6 at start
    log_std_min:           float = -3.0
    log_std_max:           float = 0.5

    # --- Output-head calibration ---
    # target_std: per-dim std of the policy mean over a batch of sampled obs.
    # max_scale:  cap on action_scale so the head can't dominate the actor.
    calib_target_std:      float = 1.5
    calib_max_scale:       float = 5.0

    model_prefix:          str   = "ppo_actor_invpen"

    def to_json(self) -> str:
        d = asdict(self)
        d["device"] = str(self.device)
        return json.dumps(d, indent=4)

    def load(self, json_data):
        data = json.loads(json_data) if isinstance(json_data, str) else dict(json_data)
        for f in self.__dataclass_fields__:
            if f in data:
                setattr(self, f, data[f])
        return self


# ──────────────────────────────────────────────────────────────────────────────
# PPO Engine
# ──────────────────────────────────────────────────────────────────────────────

class PPOEngine:
    """
    PPO engine for a stateful TWC actor and a MLP value critic.

    The actor is a ``PyUriTwc`` whose ``action_decoder`` returns the
    *mean* of a Gaussian policy.  A single learned ``log_std`` parameter
    (shape ``(action_dim,)``) is owned by the engine so it participates in
    the actor optimiser without polluting the biological model.

    The actor update uses BPTT over full collected episodes (with an
    optional burn-in prefix discarded from the loss).  The critic is
    updated on flat (non-padded) steps with an MSE loss against
    GAE-bootstrapped returns.

    Args:
        actor:            PyUriTwc instance (action_decoder → mean).
        critic:           ValueCriticInvPen instance.
        action_space:     gym.spaces.Box used for action clamping.
        actor_optimizer:  Optimiser that includes ``log_std``.
        critic_optimizer: Separate optimiser for the critic.
        config:           PPOConfig.
        device:           Torch device.
    """

    def __init__(
        self,
        actor:            PyUriTwc,
        critic:           ValueCriticInvPen,
        action_space:     gym.Space,
        actor_optimizer:  torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        config:           PPOConfig,
        device:           torch.device,
    ):
        self.actor            = actor.to(device)
        self.critic           = critic.to(device)
        self.action_space     = action_space
        self.actor_optimizer  = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.config           = config
        self.device           = device

        self.action_dim = action_space.shape[0]
        self.action_low  = torch.as_tensor(action_space.low,  device=device, dtype=torch.float32)
        self.action_high = torch.as_tensor(action_space.high, device=device, dtype=torch.float32)

        # Learned log_std — registered as a Parameter so the actor optimiser
        # updates it, but kept outside PyUriTwc to preserve the bio-model.
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), config.log_std_init, device=device)
        )
        self.action_scale = nn.Parameter(torch.ones(self.action_dim, device=device))
        self.action_bias  = nn.Parameter(torch.zeros(self.action_dim, device=device))
        self.actor_optimizer.add_param_group({"params": [self.log_std,
                                                         self.action_scale, 
                                                         self.action_bias]})

        self.total_updates = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _std(self) -> torch.Tensor:
        """Clamped std from log_std."""
        return torch.exp(
            self.log_std.clamp(self.config.log_std_min, self.config.log_std_max)
        )

    def _gaussian_log_prob(
        self, mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Log-probability of ``action`` under N(mean, std²), summed over action dims.

        Args:
            mean:   (..., action_dim)
            std:    (action_dim,) or broadcastable
            action: (..., action_dim)

        Returns:
            (...,) log-probability scalar per sample.
        """
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)

    def _entropy(self) -> torch.Tensor:
        """Entropy of the current Gaussian policy (scalar)."""
        return torch.distributions.Normal(
            torch.zeros(self.action_dim, device=self.device), self._std()
        ).entropy().sum()

    def _policy_mean(self, raw_mean: torch.Tensor) -> torch.Tensor:
        return raw_mean * self.action_scale + self.action_bias

    @torch.no_grad()
    def calibrate_output_head(
        self,
        env: gym.Env,
        n_samples: int = 512,
        target_std: float = 1.5,
        max_scale:  float = 5.0,
    ) -> dict:
        """
        Calibrate ``action_scale`` and ``action_bias`` so that, on a batch of
        observations sampled from a *real* random-action rollout, the
        post-affine policy mean has mean ≈ 0 and std ≈ ``target_std`` per
        action dim.

        Rationale: the TWC actor's raw output starts near the centre of its
        affine range (≈ 0 with very low variance), which makes the initial
        Gaussian policy a near-degenerate distribution centred at 0 — PPO
        has no gradient on the mean direction. Rescaling the readout to
        cover a meaningful fraction of the action range breaks this
        degeneracy without touching the bio-model.

        Obs source — random-action rollout. Using
        ``env.observation_space.sample()`` is wrong for unbounded Box
        spaces (e.g. InvertedPendulum-v5 declares Box(-inf, +inf)):
        Gymnasium realises that as a near-unit-Gaussian, which is far from
        the actual on-policy obs distribution and gives a misleading
        ``raw_std``. A short random-action rollout produces obs from the
        same distribution the policy will see in training.

        Scale cap. If ``raw_std`` is degenerate (the TWC sits in a
        recurrent attractor that barely depends on obs), the uncapped
        scale = target_std / raw_std becomes huge and the head dominates
        the actor: the deterministic policy becomes essentially
        obs-independent, pinned at ``scale * raw_fixed_point + bias``.
        Capping ``action_scale`` at ``max_scale`` forces PPO to train the
        TWC to produce more obs-dependent variation in its raw output
        rather than letting the head do all the work.

        Should be called once after engine construction, before training.
        """
        was_training = self.actor.training
        self.actor.eval()

        # ── Collect obs from a random-action rollout ──────────────────────
        obs_buf: list[np.ndarray] = []
        obs, _ = env.reset()
        while len(obs_buf) < n_samples:
            obs_buf.append(np.asarray(obs, dtype=np.float32))
            a = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                obs, _ = env.reset()

        obs_batch = torch.as_tensor(
            np.stack(obs_buf[:n_samples]), dtype=torch.float32
        ).to(self.device)

        self.actor.reset(batch_size=obs_batch.shape[0])
        raw = self.actor(obs_batch)                              # (N, action_dim)
        # Restore single-sample state for rollout
        self.actor.reset(batch_size=1)

        raw_mean = raw.mean(dim=0)
        raw_std  = raw.std(dim=0).clamp_min(1e-3)

        uncapped_scale = (target_std / raw_std).to(self.action_scale.device)
        new_scale = uncapped_scale.clamp(max=max_scale)
        new_bias  = (-new_scale * raw_mean).to(self.action_bias.device)
        self.action_scale.data.copy_(new_scale)
        self.action_bias.data.copy_(new_bias)

        if was_training:
            self.actor.train()

        return {
            "calib/raw_mean":       float(raw_mean.abs().mean().item()),
            "calib/raw_std":        float(raw_std.mean().item()),
            "calib/uncapped_scale": float(uncapped_scale.mean().item()),
            "calib/action_scale":   float(new_scale.mean().item()),
            "calib/action_bias":    float(new_bias.mean().item()),
            "calib/scale_capped":   float((uncapped_scale > max_scale).float().mean().item()),
        }
    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        """
        Sample one action from the current policy given a single observation.

        Args:
            obs:           (obs_dim,) numpy array.
            deterministic: If True return mean action (used during evaluation).

        Returns:
            action (np.ndarray, shape (action_dim,)),
            log_prob (float),
            value (float)  — critic estimate V(s).
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        raw_mean = self.actor(obs_t)       # (1, action_dim)  — stateful forward
        mean = self._policy_mean(raw_mean)

        if deterministic:
            action_t = mean
        else:
            std      = self._std()
            action_t = mean + torch.randn_like(mean) * std

        action_t = action_t.clamp(self.action_low, self.action_high)
        log_prob  = self._gaussian_log_prob(mean, self._std(), action_t)

        value = self.critic(obs_t).squeeze(-1)   # (1,)

        return (
            action_t.cpu().numpy()[0],
            float(log_prob.cpu().item()),
            float(value.cpu().item()),
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, batch: RolloutBatch) -> dict:
        """
        Run ``ppo_epochs`` of PPO updates on the collected rollout batch.

        Actor loss is computed via BPTT over full episode sequences
        (burn-in steps excluded from the loss).  The critic is updated
        on flat valid steps only.

        Args:
            batch: RolloutBatch from EpisodeRolloutBuffer.get_batch().

        Returns:
            dict with scalar loss metrics for logging.
        """
        cfg = self.config
        B, T_max, obs_dim = batch.obs_seq.shape
        act_dim = batch.act_seq.shape[-1]
        burn  = cfg.burn_in_length

        actor_losses, critic_losses, entropy_vals, clip_fracs = [], [], [], []
        actor_weights_grad_norms: list[float] = []

        for _ in range(cfg.ppo_epochs):
            # ── Actor update via BPTT ────────────────────────────────────────
            self.actor.train()
            self.actor_optimizer.zero_grad(set_to_none=True)

            # Run full BPTT over sequences; discard burn-in from loss
            raw_means, _ = self.actor.forward_bptt(batch.obs_seq)
            new_means = self._policy_mean(raw_means)
            # new_means: (B, T_max, action_dim)

            std = self._std()   # (action_dim,)

            # Compute log-probs for the active (non-burn-in) time steps
            act_learn  = batch.act_seq[:, burn:, :]   # (B, T', act_dim)
            mean_learn = new_means[:, burn:, :]        # (B, T', act_dim)
            adv_learn  = batch.adv_seq[:, burn:]       # (B, T')

            new_log_probs = self._gaussian_log_prob(mean_learn, std, act_learn)
            # new_log_probs: (B, T')

            old_log_probs = batch.lp_seq[:, burn:]     # (B, T')

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * adv_learn
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_learn
            actor_loss = -torch.min(surr1, surr2).mean()

            entropy_loss = -cfg.entropy_coef * self._entropy()

            (actor_loss + entropy_loss).backward()
            # Record actor weight gradient norm *before* clipping so the
            # diagnostic reflects the true signal magnitude (post-clip values
            # would be capped at max_grad_norm and hide a vanishing-grad
            # regime).
            with torch.no_grad():
                w_grad = self.actor.weights.grad
                actor_weights_grad_norms.append(
                    float(w_grad.norm().item()) if w_grad is not None else 0.0
                )
            # Clip each group independently
            norm_w  = torch.nn.utils.clip_grad_norm_([self.actor.weights] + [self.log_std, self.action_scale, self.action_bias],    max_norm=1.0)
            norm_th = torch.nn.utils.clip_grad_norm_([self.actor.thresholds], max_norm=0.5)
            norm_dc = torch.nn.utils.clip_grad_norm_([self.actor.decay],      max_norm=0.5)
            """ 
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + [self.log_std, self.action_scale, self.action_bias],
                cfg.max_grad_norm
            ) 
            """
            self.actor_optimizer.step()

            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
                clip_fracs.append(clip_frac)

            # ── Critic update (flat steps, no BPTT needed) ──────────────────
            self.critic_optimizer.zero_grad(set_to_none=True)

            values_pred = self.critic(batch.obs).squeeze(-1)   # (N,)
            critic_loss = cfg.value_loss_coef * F.mse_loss(values_pred, batch.returns)

            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
            self.critic_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_vals.append(self._entropy().item())

        self.total_updates += 1

        return {
            "loss/actor":              float(np.mean(actor_losses)),
            "loss/critic":             float(np.mean(critic_losses)),
            "loss/entropy":            float(np.mean(entropy_vals)),
            "train/clip_frac":         float(np.mean(clip_fracs)),
            "train/log_std":           float(self.log_std.mean().item()),
            "grad/actor_weights_grad_norm": float(norm_w),
            "grad/actor_threshold_grad_norm": float(norm_th),
            "grad/actor_decay_grad_norm": float(norm_dc),
            "head/action_scale":       float(self.action_scale.detach().mean().item()),
            "head/action_bias":        float(self.action_bias.detach().mean().item()),
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_policy(self, env: gym.Env, episodes: int) -> dict:
        """
        Deterministic greedy evaluation over ``episodes`` episodes.

        Returns:
            dict with mean_return, success_rate (truncated=success),
            avg_steps_success, mean_action.
        """
        self.actor.eval()

        returns, successes, steps_success, all_actions = [], [], [], []

        for _ in range(episodes):
            obs, _ = env.reset()
            self.actor.reset()
            ep_ret, done = 0.0, False

            while not done:
                action, _, _ = self.get_action(obs, deterministic=True)
                all_actions.append(float(action[0]) if np.ndim(action) > 0 else float(action))
                obs, r, terminated, truncated, _ = env.step(action)
                ep_ret += float(r)
                done = bool(terminated or truncated)

            # InvPen: success = episode survived to truncation
            success = int(truncated and not terminated)
            returns.append(ep_ret)
            successes.append(success)
            if success:
                steps_success.append(int(ep_ret))   # reward==1/step so ep_ret≈steps

        self.actor.train()

        mean_return = float(np.mean(returns)) if returns else 0.0
        success_rate = float(np.mean(successes)) if successes else 0.0
        avg_steps_success = float(np.mean(steps_success)) if steps_success else float("inf")
        mean_action = float(np.mean(all_actions)) if all_actions else 0.0

        return {
            "eval/mean_return":       mean_return,
            "eval/success_rate":      success_rate,
            "eval/avg_steps_success": avg_steps_success,
            "eval/mean_action":       mean_action,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def ppo_train(
    env:               gym.Env,
    engine:            PPOEngine,
    writer:            SummaryWriter,
    timestamp:         str,
    config:            PPOConfig,
    trial:             optuna.Trial = None,
    phi:               callable     = lambda obs: 0.0,
    skip_calibration:  bool         = False,
) -> tuple[list[float], list[float], str]:
    """
    On-policy PPO training loop for a TWC actor on InvertedPendulum-v5.

    One "update round" consists of:
        1. Collect ``episodes_per_update`` full episodes into the rollout buffer.
        2. Compute GAE advantages and run ``ppo_epochs`` PPO update passes.
        3. Clear the buffer and repeat.

    Args:
        env:       Training environment (InvertedPendulum-v5).
        engine:    Configured PPOEngine.
        writer:    TensorBoard SummaryWriter.
        timestamp: String timestamp for model file naming.
        config:    PPOConfig.
        trial:     Optional Optuna trial for pruning.
        phi:       Potential-shaping function phi(obs) -> float.

    Returns:
        (all_eval_returns, all_eval_steps, best_model_path)
    """
    cfg = config
    total_steps   = 0
    episode_count = 0
    update_round  = 0
    best_score    = -np.inf
    best_model_path = None

    all_eval_returns: list[float] = []
    all_eval_steps:   list[float] = []

    rollout_buf = EpisodeRolloutBuffer(
        gamma=cfg.gamma,
        lam=cfg.lam,
        device=torch.device(cfg.device),
        min_episodes=cfg.episodes_per_update,
    )

    # Calibrate the learnable output-head so the initial Gaussian policy
    # has non-degenerate coverage of the action range. Caller may skip this
    # if calibration was already performed (e.g. to gate trials on raw_std).
    if not skip_calibration:
        calib = engine.calibrate_output_head(
            env,
            target_std=cfg.calib_target_std,
            max_scale=cfg.calib_max_scale,
        )
        for k, v in calib.items():
            writer.add_scalar(k, v, 0)

    # Save initial weights
    initial_path = os.path.join(writer.log_dir, f"{cfg.model_prefix}_first_{timestamp}.pth")
    torch.save(engine.actor.state_dict(), initial_path)

    pbar = tqdm(total=cfg.max_train_steps, desc="Training PPO")

    while total_steps < cfg.max_train_steps:
        # ── Rollout phase ────────────────────────────────────────────────────
        engine.actor.eval()   # no dropout; eval mode for clean rollout
        rollout_buf.clear()

        for _ in range(cfg.episodes_per_update):
            if total_steps >= cfg.max_train_steps:
                break

            obs, _ = env.reset(seed=cfg.seed + episode_count)
            engine.actor.reset()

            ep_obs, ep_act, ep_lp, ep_rew, ep_val = [], [], [], [], []
            done = False
            last_value = 0.0

            while not done:
                action, log_prob, value = engine.get_action(obs)

                obs2, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)

                # Potential-based shaping
                shaped = float(reward + cfg.gamma * phi(obs2) - phi(obs))

                ep_obs.append(obs.copy())
                ep_act.append(action.copy())
                ep_lp.append(log_prob)
                ep_rew.append(shaped)
                ep_val.append(value)

                obs = obs2
                total_steps += 1
                pbar.update(1)

            # Bootstrap value for truncated episodes
            if truncated and not terminated:
                obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                        device=torch.device(cfg.device)).unsqueeze(0)
                with torch.no_grad():
                    last_value = float(engine.critic(obs_t).squeeze().item())

            rollout_buf.collect_episode(
                obs      = np.array(ep_obs,  dtype=np.float32),
                actions  = np.array(ep_act,  dtype=np.float32),
                log_probs= np.array(ep_lp,   dtype=np.float32),
                rewards  = np.array(ep_rew,  dtype=np.float32),
                values   = np.array(ep_val,  dtype=np.float32),
                last_value=last_value,
            )

            writer.add_scalar("Training/Episode_Return", sum(ep_rew), total_steps)
            writer.add_scalar("Training/Episode_Steps",  len(ep_rew), total_steps)
            episode_count += 1

        # ── Update phase ─────────────────────────────────────────────────────
        engine.actor.train()
        batch = rollout_buf.get_batch(normalize_advantages=True)
        if batch is None:
            continue

        metrics = engine.update(batch)
        update_round += 1

        for k, v in metrics.items():
            writer.add_scalar(k, v, total_steps)

        # ── Evaluation ───────────────────────────────────────────────────────
        if update_round % cfg.eval_interval_updates == 0:
            eval_metrics = engine.evaluate_policy(env, episodes=cfg.eval_episodes)

            for k, v in eval_metrics.items():
                writer.add_scalar(k, v, total_steps)

            eval_ret   = eval_metrics["eval/mean_return"]
            eval_steps = eval_metrics["eval/avg_steps_success"]
            all_eval_returns.append(eval_ret)
            all_eval_steps.append(eval_steps)

            eval_idx = len(all_eval_returns)

            if eval_ret > best_score:
                best_score = eval_ret
                path = os.path.join(
                    writer.log_dir, f"{cfg.model_prefix}_best_{timestamp}.pth"
                )
                torch.save(engine.actor.state_dict(), path)
                best_model_path = path
                tqdm.write(f"[update {update_round}] new best return {eval_ret:.2f} → {path}")

            if trial is not None:
                trial.report(best_score, step=eval_idx)
                if trial.should_prune():
                    tqdm.write(f"Trial {trial.number} pruned at update {update_round}")
                    pbar.close()
                    raise optuna.TrialPruned()

    pbar.close()

    # Final checkpoint
    final_path = os.path.join(writer.log_dir, f"{cfg.model_prefix}_final_{timestamp}.pth")
    torch.save(engine.actor.state_dict(), final_path)
    tqdm.write(f"Final model → {final_path}")

    return all_eval_returns, all_eval_steps, best_model_path