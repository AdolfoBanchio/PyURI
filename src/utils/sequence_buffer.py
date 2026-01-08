import numpy as np
import torch

class SequenceBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.episodes = [] 
        self.total_transitions = 0
        self._init_current_episode()

    def _init_current_episode(self):
        self.curr_obs = []
        self.curr_act = []
        self.curr_rew = []
        self.curr_nobs = []
        self.curr_term = []
        self.curr_trunc = []

    def store(self, obs, action, reward, next_obs, terminated, truncated):
        self.curr_obs.append(obs)
        self.curr_act.append(action)
        self.curr_rew.append(reward)
        self.curr_nobs.append(next_obs)
        self.curr_term.append(terminated)
        self.curr_trunc.append(truncated)

        if terminated or truncated:
            episode = {
                "obs": np.array(self.curr_obs, dtype=np.float32),
                "action": np.array(self.curr_act, dtype=np.float32),
                "reward": np.array(self.curr_rew, dtype=np.float32).reshape(-1, 1),
                "next_obs": np.array(self.curr_nobs, dtype=np.float32),
                "terminated": np.array(self.curr_term, dtype=np.float32).reshape(-1, 1),
                "truncated": np.array(self.curr_trunc, dtype=np.float32).reshape(-1, 1)
            }
            self.episodes.append(episode)
            self.total_transitions += len(self.curr_obs)
            self._init_current_episode()

            while self.total_transitions > self.capacity:
                old_ep = self.episodes.pop(0)
                self.total_transitions -= len(old_ep["obs"])

    def sample(self, batch_size: int, sequence_length: int):
        # Filtrar episodios que tengan al menos (sequence_length) transiciones
        # Optimizamos: solo recalculamos si es estrictamente necesario o usamos un cache
        valid_episodes = [ep for ep in self.episodes if len(ep["obs"]) >= sequence_length]
        
        if not valid_episodes:
            return None

        # Pesos para el muestreo: episodios más largos tienen más probabilidad (opcional)
        lengths = np.array([len(ep["obs"]) - sequence_length + 1 for ep in valid_episodes])
        probs = lengths / lengths.sum()

        batch = {k: [] for k in ["obs", "action", "reward", "next_obs", "terminated", "truncated"]}
        
        sampled_indices = np.random.choice(len(valid_episodes), size=batch_size, p=probs)
        
        for idx in sampled_indices:
            ep = valid_episodes[idx]
            max_start = len(ep["obs"]) - sequence_length
            start = np.random.randint(0, max_start + 1)
            end = start + sequence_length

            for key in batch.keys():
                batch[key].append(ep[key][start:end])

        # Stack y envío a device en una sola operación
        return {k: torch.as_tensor(np.stack(v), device=self.device) for k, v in batch.items()}
