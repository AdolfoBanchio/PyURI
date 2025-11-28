import torch
import numpy as np

class PrioritizedSequenceReplayBuffer:
    """
    Prioritized *Sequence* Experience Replay (PSER-style) buffer
    para uso con BPTT y TD3.
    """
    def __init__(
        self,
        obs_shape,
        act_shape,
        capacity: int,
        alpha: float = 0.6,       # grado de priorización (PER/PSER)
        beta_start: float = 0.4,  # IS weights initial exponent
        beta_frames: int = 100_000,  # pasos de actualización para annealing β→1
        epsilon: float = 1e-6,    # ε para p_i = |δ| + ε
        rho: float = 0.4,         # coeficiente de decaimiento de PSER
        eta: float = 0.7,         # parámetro anti "priority collapse"
        seed: int = 42,
    ):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = max(1, int(beta_frames))
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / self.beta_frames
        self.epsilon = epsilon
        self.rho = rho
        self.eta = eta

        self.rng = np.random.RandomState(seed)

        # Buffers
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.act_buf = np.zeros((capacity, *act_shape), dtype=np.float32)
        self.rew_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.bool_)

        # Episodios para respetar límites de secuencia
        self.episode_id_buf = np.full((capacity,), -1, dtype=np.int64)
        self.current_episode = 0

        # Prioridades por transición
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0  # se actualiza conforme se aprende

        # Punteros
        self.ptr = 0
        self.size = 0

        # Ventana W para decaimiento de PSER (ecuación (9))
        if 0.0 < self.rho < 1.0:
            self.window = int(np.floor(np.log(0.01) / np.log(self.rho)))
        else:
            self.window = 0  # sin decaimiento si rho no es válido

    # ------------------------------------------------------------------ #
    # Almacenamiento
    # ------------------------------------------------------------------ #
    def store(self, obs, act, rew, next_obs, terminated: bool, truncated: bool):
        """
        Guarda una transición en el buffer.

        obs, next_obs: np.array o lista convertible a np.array
        act: np.array o lista
        rew: float
        terminated, truncated: bools del entorno Gymnasium
        """
        done = bool(terminated or truncated)

        idx = self.ptr

        self.obs_buf[idx] = np.asarray(obs, dtype=np.float32)
        self.next_obs_buf[idx] = np.asarray(next_obs, dtype=np.float32)
        self.act_buf[idx] = np.asarray(act, dtype=np.float32)
        self.rew_buf[idx] = float(rew)
        self.done_buf[idx] = done

        # Episodio actual
        self.episode_id_buf[idx] = self.current_episode

        # Prioridad inicial: max_priority (como en PER/PSER)
        self.priorities[idx] = self.max_priority

        # Actualizar puntero circular
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        # Si el episodio terminó, incrementamos ID
        if done:
            self.current_episode += 1

    # ------------------------------------------------------------------ #
    # Muestreo de secuencias
    # ------------------------------------------------------------------ #
    def _compute_valid_end_indices(self, seq_len: int):
        """
        Devuelve todos los índices i que pueden usarse como final
        de una secuencia de longitud seq_len sin cruzar episodios.
        """
        if self.size < seq_len:
            return np.array([], dtype=np.int64)

        # Solo índices dentro de [0, size-1]
        indices = np.arange(self.size, dtype=np.int64)

        # i es válido si:
        # - i >= seq_len-1
        # - episodio_id constante en [i - seq_len + 1, ..., i]
        valid_mask = np.zeros(self.size, dtype=bool)
        for i in range(seq_len - 1, self.size):
            start = i - seq_len + 1
            ep = self.episode_id_buf[i]
            if ep < 0:
                continue
            if np.all(self.episode_id_buf[start : i + 1] == ep):
                valid_mask[i] = True

        return indices[valid_mask]

    def sample(self, batch_size: int, seq_len: int, device: torch.device):
        """
        Devuelve:
            seq_batch: dict con tensores (B, L, ...)
                - 'obs'
                - 'action'
                - 'reward'
                - 'next_obs'
                - 'done'
            is_weights: tensor (B, 1) con importance-sampling weights
            idxs: np.array (B,) con índices finales de secuencia

        Diseñado para:
            seq_batch, is_weights, idxs = buffer.sample(...)
        y luego en tu motor:
            actor_loss, critic_loss, td_errors = engine.update_step_bptt(
                seq_batch, burn_in_length, is_weights
            )
        """
        assert seq_len >= 1
        assert batch_size >= 1

        valid_end_indices = self._compute_valid_end_indices(seq_len)
        if len(valid_end_indices) == 0:
            raise RuntimeError(
                f"No hay suficientes secuencias válidas (seq_len={seq_len}, size={self.size})."
            )

        # Prioridades solo sobre índices válidos (como PSER sobre transiciones)
        valid_prios = self.priorities[valid_end_indices]
        if valid_prios.sum() == 0:
            # Evitar NaN si todo es cero
            valid_prios = np.ones_like(valid_prios, dtype=np.float32)

        scaled_prios = valid_prios ** self.alpha
        probs = scaled_prios / scaled_prios.sum()

        # Sampleo de índices finales de secuencia
        idxs = self.rng.choice(
            valid_end_indices, size=batch_size, replace=len(valid_end_indices) < batch_size, p=probs
        )

        # Probabilidades de los índices seleccionados
        prob_dict = {int(i): float(p) for i, p in zip(valid_end_indices, probs)}
        p_samples = np.array([prob_dict[int(i)] for i in idxs], dtype=np.float32)

        # Importance sampling weights (ecuación (11))
        N = len(valid_end_indices)
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (N * p_samples) ** (-self.beta)
        weights /= weights.max()  # normalizar

        # Construir batch de secuencias
        B, L = batch_size, seq_len
        obs_batch = np.zeros((B, L, *self.obs_buf.shape[1:]), dtype=np.float32)
        next_obs_batch = np.zeros_like(obs_batch)
        act_batch = np.zeros((B, L, *self.act_buf.shape[1:]), dtype=np.float32)
        rew_batch = np.zeros((B, L), dtype=np.float32)
        done_batch = np.zeros((B, L), dtype=np.float32)

        for b, end_idx in enumerate(idxs):
            end = int(end_idx)
            start = end - seq_len + 1
            obs_batch[b] = self.obs_buf[start : end + 1]
            next_obs_batch[b] = self.next_obs_buf[start : end + 1]
            act_batch[b] = self.act_buf[start : end + 1]
            rew_batch[b] = self.rew_buf[start : end + 1]
            done_batch[b] = self.done_buf[start : end + 1].astype(np.float32)

        # Convertir a tensores
        seq_batch = {
            "obs": torch.as_tensor(obs_batch, device=device, dtype=torch.float32),
            "next_obs": torch.as_tensor(next_obs_batch, device=device, dtype=torch.float32),
            "action": torch.as_tensor(act_batch, device=device, dtype=torch.float32),
            "reward": torch.as_tensor(rew_batch, device=device, dtype=torch.float32),
            "done": torch.as_tensor(done_batch, device=device, dtype=torch.float32),
        }

        is_weights = torch.as_tensor(weights, device=device, dtype=torch.float32).unsqueeze(-1)

        return seq_batch, is_weights, idxs

    # ------------------------------------------------------------------ #
    # Actualización de prioridades (PSER)
    # ------------------------------------------------------------------ #
    def update_priorities(self, idxs, td_errors):
        """
        Actualiza las prioridades de los índices finales de secuencias
        y propaga prioridad hacia atrás en el episodio como PSER.

        idxs: np.array o tensor 1D de índices finales (dentro del buffer).
        td_errors: np.array o tensor 1D de TD-errors (mismo orden que idxs).

        PSER:
            1) prioridad local:
                p_i <- max(|δ| + ε, η * p_i_old)     (ecuación (10))
            2) backward decay hacia transiciones previas del mismo episodio
               usando coeficiente ρ y ventana W (ecuaciones (6)-(9)).
        """
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.detach().cpu().numpy()
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()

        idxs = np.asarray(idxs, dtype=np.int64)
        td_errors = np.asarray(td_errors, dtype=np.float32)
        assert idxs.shape == td_errors.shape

        for idx, delta in zip(idxs, td_errors):
            idx = int(idx)
            if idx < 0 or idx >= self.size:
                continue

            abs_delta = abs(float(delta))
            old_p = float(self.priorities[idx])

            # Paso 1: prioridad local con η (anti priority collapse)
            new_p = max(abs_delta + self.epsilon, self.eta * old_p)
            self.priorities[idx] = new_p

            # Mantener max_priority actualizado
            if new_p > self.max_priority:
                self.max_priority = new_p

            # Paso 2: decaimiento hacia atrás dentro del episodio (MAX variant)
            if self.window <= 0 or not (0.0 < self.rho < 1.0):
                continue

            ep_id = self.episode_id_buf[idx]
            if ep_id < 0:
                continue

            j = idx - 1
            k = 1
            while j >= 0 and k <= self.window:
                if self.episode_id_buf[j] != ep_id:
                    break

                decayed_p = new_p * (self.rho ** k)
                if decayed_p <= self.priorities[j]:
                    # ya tiene prioridad mayor, no hace falta seguir
                    break

                self.priorities[j] = decayed_p
                if decayed_p > self.max_priority:
                    self.max_priority = decayed_p

                j -= 1
                k += 1