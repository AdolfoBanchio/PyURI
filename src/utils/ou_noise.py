import numpy as np

class OUNoise:
    def __init__(self, action_dim, config):
        self.action_dim = action_dim
        self.theta = config.ou_theta
        self.sigma_init = config.ou_sigma_init
        self.sigma_end = config.ou_sigma_end
        self.decay_steps = config.ou_sigma_decay_steps
        self.sigma = self.sigma_init
        self.state = np.zeros(self.action_dim)
        self.reset()

    def reset(self):
        self.state = np.zeros(self.action_dim)

    def update(self, total_steps):
        # Decaimiento lineal de sigma
        fraction = min(float(total_steps) / self.decay_steps, 1.0)
        self.sigma = self.sigma_init + fraction * (self.sigma_end - self.sigma_init)

    def noise(self):
        x = self.state
        dx = self.theta * (-x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state