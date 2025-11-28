import numpy as np

import numpy as np

class OUNoise:
    """
    Ornstein–Uhlenbeck noise para exploración correlada en TD3.
    
    Métodos externos:
        - noise(): devuelve un valor de ruido
        - update(total_steps): actualiza sigma según el progreso del entrenamiento
        
    Parámetros típicos:
        mu: media a la que “vuelve” el ruido
        theta: velocidad de retorno (más alto = más suave el ruido)
        sigma_init: desviación inicial del ruido
        sigma_min: desviación mínima al final del entrenamiento
        decay_steps: pasos totales para decaer sigma_init → sigma_min
        dt: paso de tiempo del proceso OU (normalmente 1)
    """

    def __init__(
        self,
        size,
        mu=0.0,
        theta=0.15,
        sigma_init=0.2,
        sigma_min=0.05,
        decay_steps=300_000,
        dt=1.0,
        seed=None,
    ):
        self.size = size if isinstance(size, tuple) else (size,)
        self.mu = mu
        self.theta = theta
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.decay_steps = max(decay_steps, 1)
        self.dt = dt
        
        self.sigma = sigma_init
        self.state = np.ones(self.size) * self.mu
        
        self.rng = np.random.RandomState(seed)

    # ----------------------------------------------------------
    def reset(self):
        """Reinicia el estado interno del proceso OU"""
        self.state = np.ones(self.size) * self.mu

    # ----------------------------------------------------------
    def update(self, total_steps):
        """
        Actualiza sigma en función de la proporción de entrenamiento completada.
        Puedes usar decay lineal o exponencial.
        """
        frac = min(total_steps / self.decay_steps, 1.0)

        # Decaimiento lineal clásico: sigma = sigma_init → sigma_min
        self.sigma = self.sigma_init - frac * (self.sigma_init - self.sigma_min)

        # Alternativa :
        # Decaimiento exponencial
        # self.sigma = self.sigma_min + (self.sigma_init - self.sigma_min) * np.exp(-3 * frac)

    # ----------------------------------------------------------
    def noise(self):
        """
        Genera un paso del proceso OU.
        dx = theta (mu - x) dt + sigma sqrt(dt) * N(0,1)
        """
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * self.rng.randn(*self.size)
        self.state = x + dx
        return self.state.copy()  
