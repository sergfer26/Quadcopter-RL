import numpy as np
from env.equations import penalty, angles2rotation
from env.params import STATE_PARAMS


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
    # if np.linalg.norm(x[3:6]) < 2.5:
    #     reward = 10.0
    return reward - penalty(x, u, i)



def get_sparse_reward(x: np.ndarray, u: np.ndarray, i: int, 
                      x_target: np.ndarray = None) -> float:
    if not isinstance(x_target, np.ndarray):
        x_target = np.zeros_like(x)

    mat = angles2rotation(x[9:], flatten=False)

    r = 100 - 100 * np.tanh(np.linalg.norm(x[3:6] - x_target[3:6])) ** 2
    r += 10 - 10 * np.tanh(np.linalg.norm(np.identity(3) - mat)) ** 2

    limits = np.array(
        [eval(STATE_PARAMS['$x$']), eval(STATE_PARAMS['$y$']), eval(STATE_PARAMS['$z$'])]
        )
    position_bool = np.logical_and(-limits <= x[3:6], x[3:6] <= limits)
    if not position_bool.all():
        r -= 100
    return r
