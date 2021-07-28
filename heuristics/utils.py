import numpy as np


def distance(state1, state2):
    return np.sum((fs2numpy(state1) - fs2numpy(state2))**2)


def fs2numpy(state):
    return np.array([state.x, state.z])
