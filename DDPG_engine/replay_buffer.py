import numpy as np
import torch

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