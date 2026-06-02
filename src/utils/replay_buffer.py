import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RolloutBatch:
    """Flattened (B*T, ...) tensors ready for PPO update."""
    obs:        torch.Tensor   # (B*T, obs_dim)
    action:     torch.Tensor   # (B*T, act_dim)
    log_prob:   torch.Tensor   # (B*T,)
    advantage:  torch.Tensor   # (B*T,)
    returns:    torch.Tensor   # (B*T,)
    # Sequence view — kept for BPTT actor update
    obs_seq:    torch.Tensor   # (B, T, obs_dim)
    act_seq:    torch.Tensor   # (B, T, act_dim)
    lp_seq:     torch.Tensor   # (B, T)
    adv_seq:    torch.Tensor   # (B, T)
    ret_seq:    torch.Tensor   # (B, T)


class EpisodeRolloutBuffer:
    """
    On-policy rollout buffer that stores complete episodes and computes
    GAE(λ) advantages.  Designed for a stateful TWC actor trained with BPTT.

    Each call to ``collect_episode`` appends one episode.  Call ``get_batch``
    to retrieve a padded, tensorised batch once enough episodes are stored,
    then ``clear`` before the next rollout phase.

    Args:
        gamma:      Discount factor.
        lam:        GAE lambda.
        device:     Torch device for output tensors.
        min_episodes: Minimum episodes before ``get_batch`` returns data.
    """

    def __init__(
        self,
        gamma: float,
        lam: float,
        device: torch.device,
        min_episodes: int = 1,
    ):
        self.gamma = float(gamma)
        self.lam   = float(lam)
        self.device = device
        self.min_episodes = int(min_episodes)
        self._episodes: list[dict] = []

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect_episode(
        self,
        obs:      np.ndarray,   # (T, obs_dim)
        actions:  np.ndarray,   # (T, act_dim)
        log_probs: np.ndarray,  # (T,)
        rewards:  np.ndarray,   # (T,)
        values:   np.ndarray,   # (T,)   V(s_t) from critic
        last_value: float,      # V(s_T) — 0.0 if terminated, critic(s_T) if truncated
    ):
        """
        Store one complete episode and compute its GAE(λ) advantages.

        Args:
            obs:        Raw observations, shape (T, obs_dim).
            actions:    Actions taken, shape (T, act_dim).
            log_probs:  Log-probabilities of those actions, shape (T,).
            rewards:    Rewards received, shape (T,).
            values:     Critic value estimates V(s_t), shape (T,).
            last_value: Bootstrap value for the step after episode end.
                        Pass 0.0 on ``terminated``, critic(s_T) on ``truncated``.
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.lam * gae
            advantages[t] = gae

        returns = advantages + values

        self._episodes.append({
            "obs":       obs.astype(np.float32),
            "action":    actions.astype(np.float32),
            "log_prob":  log_probs.astype(np.float32),
            "advantage": advantages,
            "returns":   returns.astype(np.float32),
        })

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def ready(self) -> bool:
        """True when enough episodes have been collected."""
        return len(self._episodes) >= self.min_episodes

    def get_batch(self, normalize_advantages: bool = True) -> Optional[RolloutBatch]:
        """
        Pad episodes to the same length, stack into tensors, return a
        ``RolloutBatch``.

        Shorter episodes are zero-padded on the right; the padding does not
        contribute to any loss because the PPO update masks on sequence length.
        Advantage normalisation is computed only over valid (non-padded) steps.

        Returns:
            RolloutBatch or None if not enough episodes.
        """
        if not self.ready():
            return None

        eps = self._episodes
        T_max = max(len(ep["obs"]) for ep in eps)

        obs_dim = eps[0]["obs"].shape[1]
        act_dim = eps[0]["action"].shape[1]
        B = len(eps)

        obs_arr  = np.zeros((B, T_max, obs_dim),  dtype=np.float32)
        act_arr  = np.zeros((B, T_max, act_dim),  dtype=np.float32)
        lp_arr   = np.zeros((B, T_max),            dtype=np.float32)
        adv_arr  = np.zeros((B, T_max),            dtype=np.float32)
        ret_arr  = np.zeros((B, T_max),            dtype=np.float32)
        mask_arr = np.zeros((B, T_max),            dtype=np.float32)

        for i, ep in enumerate(eps):
            T = len(ep["obs"])
            obs_arr[i, :T]  = ep["obs"]
            act_arr[i, :T]  = ep["action"]
            lp_arr[i,  :T]  = ep["log_prob"]
            adv_arr[i, :T]  = ep["advantage"]
            ret_arr[i, :T]  = ep["returns"]
            mask_arr[i, :T] = 1.0

        if normalize_advantages:
            valid = mask_arr.astype(bool)
            mu  = adv_arr[valid].mean()
            std = adv_arr[valid].std() + 1e-8
            adv_arr[valid] = (adv_arr[valid] - mu) / std

        def _t(x):
            return torch.as_tensor(x, device=self.device)

        obs_seq = _t(obs_arr)   # (B, T_max, obs_dim)
        act_seq = _t(act_arr)
        lp_seq  = _t(lp_arr)
        adv_seq = _t(adv_arr)
        ret_seq = _t(ret_arr)
        mask    = _t(mask_arr)  # (B, T_max)  — kept for caller if needed

        # Flat views (only valid steps matter for critic MSE and ratio clip)
        flat_mask = mask.bool().reshape(-1)
        obs_flat  = obs_seq.reshape(-1, obs_dim)[flat_mask]
        act_flat  = act_seq.reshape(-1, act_dim)[flat_mask]
        lp_flat   = lp_seq.reshape(-1)[flat_mask]
        adv_flat  = adv_seq.reshape(-1)[flat_mask]
        ret_flat  = ret_seq.reshape(-1)[flat_mask]

        return RolloutBatch(
            obs=obs_flat, action=act_flat, log_prob=lp_flat,
            advantage=adv_flat, returns=ret_flat,
            obs_seq=obs_seq, act_seq=act_seq, lp_seq=lp_seq,
            adv_seq=adv_seq, ret_seq=ret_seq,
        )

    def clear(self):
        """Discard all stored episodes (call after each PPO update)."""
        self._episodes.clear()

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def total_steps(self) -> int:
        """Total transition count currently stored."""
        return sum(len(ep["obs"]) for ep in self._episodes)