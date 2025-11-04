import numpy as np

class OUNoise:
    def __init__(self,
                 action_dimension,
                 mu=0, 
                 theta=0.15, 
                 sigma=0.2,
                 sigma_end=0.05,
                 sigma_decay_epis=150):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.sigma_start = sigma
        self.sigma_end = sigma_end
        self.sigma_decay = sigma_decay_epis
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
    def update_sigma(self, ep):
        frac = min(1.0, ep / self.sigma_decay)
        self.sigma = self.sigma_start + (self.sigma_end - self.sigma_start) * frac