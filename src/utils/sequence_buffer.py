import numpy as np
import torch
import random
from collections import deque

class SequenceBuffer:
    """
    A replay buffer that stores and samples sequences of transitions for BPTT.
    
    This buffer stores entire episodes. When sampling, it pulls a fixed-length
    sequence from a random episode, ensuring that the sequence never
    crosses an episode boundary.
    """
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: The maximum number of *transitions* (not episodes) to store.
        """
        # Buffer of episodes. Each episode is a dict of numpy arrays.
        self.episodes = deque()
        
        self.capacity = int(capacity)
        self.total_transitions = 0
        
        # Temporary buffer for the episode currently being collected
        self._init_current_episode()

    @property
    def size(self) -> int:
        """Returns the total number of transitions stored in the buffer."""
        return self.total_transitions

    def _init_current_episode(self):
        """Resets the temporary episode buffer."""
        self.current_episode = {
            "obs": [],
            "action": [],
            "reward": [],
            "next_obs": [],
            "done": [],
            "truncated": [],
        }

    def store(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool, truncated: bool):
        """
        Stores a single transition. If 'done' or 'truncated' is True,
        the current episode is "flushed" to the main buffer.
        
        Note: We use 'done' to mean terminal state, 'truncated' is just
        a time limit, but for sampling, both mark the end of a valid sequence.
        """
        self.current_episode["obs"].append(obs)
        self.current_episode["action"].append(action)
        self.current_episode["reward"].append(np.array([reward])) # Store as (1,)
        self.current_episode["next_obs"].append(next_obs)
        self.current_episode["done"].append(np.array([done])) # Store as (1,)
        self.current_episode['truncated'].append(np.array([truncated])) # Store as (1,)

        # An episode ends if it's done (terminal) OR truncated (time limit)
        if done or truncated:
            self._flush_current_episode()

    def _flush_current_episode(self):
        """
        Converts the temporary episode buffer (lists) into a
        dict of numpy arrays and adds it to the main episode deque.
        """
        ep_len = len(self.current_episode["obs"])
        
        # Don't store empty episodes
        if ep_len == 0:
            return

        # Convert all lists to stacked numpy arrays
        flushed_episode = {}
        for key in self.current_episode.keys():
            flushed_episode[key] = np.stack(self.current_episode[key])
            
        # Add to the buffer
        self.episodes.append(flushed_episode)
        self.total_transitions += ep_len
        
        # Evict old episodes if we are over capacity
        while self.total_transitions > self.capacity:
            evicted_episode = self.episodes.popleft()
            self.total_transitions -= len(evicted_episode["obs"])
            
        # Reset the temporary buffer
        self._init_current_episode()

    def sample(self, batch_size: int, sequence_length: int, device: torch.device) -> dict:
        """
        Samples a batch of transition sequences for BPTT.

        Args:
            batch_size: The number of sequences to sample.
            sequence_length: The length of each sequence (e.g., 40 for burn-in + 40 for train).
            device: The torch device to send the tensors to.

        Returns:
            A dictionary of tensors, each with shape (batch_size, sequence_length, *).
        """
        
        # 1. Find all episodes that are long enough to sample from
        valid_episodes = [ep for ep in self.episodes if len(ep["obs"]) >= sequence_length]
        
        if not valid_episodes:
            raise ValueError(f"Not enough data to sample sequences. "
                             f"Need episodes >= {sequence_length} steps, but found none. "
                             f"Total transitions: {self.total_transitions}")

        batch_seq = {key: [] for key in self.current_episode.keys()}

        # 2. Sample 'batch_size' sequences
        for _ in range(batch_size):
            # Pick a random valid episode
            ep = random.choice(valid_episodes)
            
            # Pick a random valid start index within that episode
            max_start_idx = len(ep["obs"]) - sequence_length
            start = np.random.randint(0, max_start_idx + 1)
            end = start + sequence_length
            
            # Slice the episode and add to our batch
            for key in batch_seq.keys():
                batch_seq[key].append(ep[key][start:end])

        # 3. Stack, convert to tensors, and send to device
        tensor_batch = {}
        for key, data_list in batch_seq.items():
            # Stack all sequences in the batch
            stacked_data = np.stack(data_list) # Shape: (batch_size, sequence_length, *)
            
            # Reshape rewards and dones from (B, L, 1) to (B, L)
            if key in ['reward', 'done'] and stacked_data.ndim == 3 and stacked_data.shape[2] == 1:
                stacked_data = stacked_data.reshape(batch_size, sequence_length)
                
            tensor_batch[key] = torch.tensor(stacked_data, dtype=torch.float32, device=device)

        return tensor_batch
