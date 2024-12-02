import numpy as np
import gym

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    '''
    Ornstein-Ulhenbeck Process
    '''
    def __init__(self, action_space = None, 
                 mu: float = 0.0,
                 theta: float = 0.15,
                 max_sigma: float = 0.9,
                 min_sigma: float = 0.05,
                 decay_period: float = 1e4
                 ):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        if isinstance(action_space, gym.spaces.Box):
            self.set_action_space(action_space)

    def set_action_space(self, action_space) -> None:
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self) -> None:
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self) -> np.ndarray:
        '''
        Evolve the state

        Returns
        -------
        state: np.ndarray
            Evolved state
        ''' 
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.action_dim)
        self.state = x + dx
        return np.clip(self.state, self.low, self.high)

    def get_action(self, action: np.ndarray, t: float = 0.0) -> np.ndarray:
        '''
        Get the action with noise

        Args
        ----
        action: np.ndarray
            Action to add noise to
        t: float
            Time step

        Returns
        -------
        action: np.ndarray
            Action with noise
        '''
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - \
            (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)