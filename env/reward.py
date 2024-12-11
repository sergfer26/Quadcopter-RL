import numpy as np
from env.equations import penalty


def get_reward(x: np.ndarray, u: np.ndarray, i: int):
    '''
    Get the reward of the system.
    
    Args
    ----
    x: np.ndarray
        State of the system.
    u: np.ndarray
        Action of the system.
    i: int
        Time step.

    Returns
    -------
    reward: float
        Reward of the system.
    '''
    reward = 0.0
    if np.linalg.norm(x[3:6]) < 2.5:
        reward = 10.0
    return reward - penalty(x, u, i)



def get_sparse_reward(x: np.ndarray, u: np.ndarry, i: int, 
                      p_1: np.ndarray = np.array([10, 10, 20]), 
                      x_target: np.ndarray = None):
    if not isinstance(x_target, np.ndarray):
        x_target = np.zeros_like(x)

    delta_weighted = p_1 * (x[3:6] - x_target[3:6])
    r = np.tanh(1 - np.linalg.norm(delta_weighted) ** 2) 
    return max(0, r)
