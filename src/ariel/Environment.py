import numpy as np


def getRandomObservation(env):
    if env=="MouCarCon":
        randomX = np.random.uniform(-1.2, 0.6,1)
        randomV = np.random.uniform(-0.07, 0.07,1)
    return randomX,randomV