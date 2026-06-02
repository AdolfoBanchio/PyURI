import numpy as np


class GaussianNoise:
    """
    Uncorrelated Gaussian exploration noise with a linear sigma schedule.

    Same interface as ``OUNoise`` so it's drop-in usable inside ``td3_train``:
    ``reset()``, ``update(total_steps)``, ``noise()``.

    Config attributes expected:
        - gaussian_sigma_init        (float)
        - gaussian_sigma_end         (float)
        - gaussian_sigma_decay_steps (int)
    """

    def __init__(self, action_dim, config):
        self.action_dim   = action_dim
        self.sigma_init   = float(config.gaussian_sigma_init)
        self.sigma_end    = float(config.gaussian_sigma_end)
        self.decay_steps  = int(config.gaussian_sigma_decay_steps)
        self.sigma        = self.sigma_init

    def reset(self):
        # No internal state — kept for API parity with OUNoise.
        pass

    def update(self, total_steps):
        fraction   = min(float(total_steps) / self.decay_steps, 1.0)
        self.sigma = self.sigma_init + fraction * (self.sigma_end - self.sigma_init)

    def noise(self):
        return self.sigma * np.random.randn(self.action_dim)
