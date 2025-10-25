import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=1e6):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1),       dtype=np.float32)
        self.obs2= np.zeros((size, obs_dim), dtype=np.float32)
        self.terminated = np.zeros((size,1), dtype=np.float32)
        self.truncated = np.zeros((size,1), dtype=np.float32)
        self.max_size, self.ptr, self.size = int(size), 0, 0
        self.keep_first = int(20_000)

    def store(self, s, a, r, s2, ter, trunc):
        self.obs[self.ptr]  = s
        self.act[self.ptr]  = a
        self.rew[self.ptr]  = r
        self.obs2[self.ptr] = s2
        self.terminated[self.ptr] = ter
        self.truncated[self.ptr] = trunc
        self.ptr = (self.ptr + 1) % self.max_size
        
        if self.ptr == 0: 
            self.ptr = self.keep_first # evict first random samples
        
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        # before we have keep_first samples, draw from [0, size)
        if self.size <= self.keep_first:
            low, high = 0, self.size
        else:
            # after that, always draw from [keep_first, size)
            low, high = self.keep_first, self.size

        idx = np.random.randint(low, high, size=batch_size)
        batch = dict(
            obs  = torch.as_tensor(self.obs[idx],  device=device),
            act  = torch.as_tensor(self.act[idx],  device=device),
            rew  = torch.as_tensor(self.rew[idx],  device=device),
            obs2 = torch.as_tensor(self.obs2[idx], device=device),
            terminated = torch.as_tensor(self.terminated[idx], device=device),
            truncated = torch.as_tensor(self.truncated[idx], device=device),
        )
        
        return batch
    