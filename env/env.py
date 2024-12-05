import gym
import numpy as np
from copy import copy
from gym.core import Env
from env.equations import f
from scipy.integrate import odeint
from spinup.algos.pytorch.ddpg.noise import OUNoise
from env.constants import W0, omega_0, C
from env.params import ENV_PARAMS, STATE_PARAMS
from env.equations import angles2rotation, penalty

omega0_per = .60
HIGH_ACT = omega_0 * omega0_per  # 60 #Velocidad maxima de los motores 150
LOW_ACT = - omega_0 * omega0_per
LOW_OBS = np.array([- eval(v) for v in STATE_PARAMS.values()])
HIGH_OBS = np.array([eval(v) for v in STATE_PARAMS.values()])
DT = ENV_PARAMS['dt']
STEPS = ENV_PARAMS['STEPS']


def get_reward(x, u, i):
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
    return reward-penalty(x, u, i)


class QuadcopterEnv(gym.Env):
    def __init__(self, 
                 check_contained: bool = False,
                 is_training: bool = False, 
                 reward: callable = lambda x, u, i : get_reward(x, u, i),
                 low_obs: np.ndarray = LOW_OBS,
                 high_obs: np.ndarray = HIGH_OBS,
                 low_act: np.ndarray = LOW_ACT * np.ones(4),
                 high_act: np.ndarray = HIGH_ACT * np.ones(4),
                 noise: OUNoise = None
                 ) -> None:
        '''
        Initialize the Quadcopter environment.

        Args
        ----
        check_contained: bool
            Whether to check if the state is contained in the observation space.
        is_training: bool
            Whether the environment is in training mode.
        reward: callable
            Function to compute the reward.
        low_obs: np.ndarray
            Lower bounds of the observation space.
        high_obs: np.ndarray
            Upper bounds of the observation space.
        low_act: np.ndarray
            Lower bounds of the action space.
        high_act: np.ndarray
            Upper bounds of the action space.
        '''
        super().__init__()
        self.action_space = gym.spaces.Box(low=low_act,
                                           high=high_act, 
                                           dtype=np.float32)
        self._observation_space = gym.spaces.Box(low=low_obs, 
                                                high=high_obs, 
                                                dtype=np.float32)
        self.observation_space = copy(self._observation_space)
        self.state = self.reset()  # estado interno del ambiente
        self.set_time(STEPS, DT)
        self.reward = reward
        self.check_contained = check_contained
        self.is_training = is_training
        self.training_noise = noise
        if isinstance(noise, OUNoise):
            noise.set_action_space(self.action_space)

    def is_contained(self, state):
        '''
        Check if the state is contained in the observation space.

        Args
        ----
        state: np.ndarray
            State of the system.

        Returns
        -------
        contained: bool
            Whether the state is contained in the observation space.
        '''
        x = state[3:6]
        high = self._observation_space.high[3:6]
        low = self._observation_space.low[3:6]
        aux = np.logical_and(low <= x, x <= high)
        return aux.all()

    def is_done(self):
        '''
        Check if the episode is done.

        Returns
        -------
        done: bool
            Whether the episode is done.
        '''
        done = False
        if self.i == self.steps-1:  # Si se te acabo el tiempo
            done = True
        return done
        
    def step(self, action):
        '''
        Step the environment.  

        Args
        ----
        action: np.ndarray
            Action of the system.

        Returns
        -------
        state: np.ndarray
            State of the system.
        reward: float
            Reward of the system.
        done: bool
            Whether the episode is done.
        info: dict
            Additional information.
        '''
        if self.training_noise is not None:
            action = self.training_noise.get_action(action, self.i)
        w1, w2, w3, w4 = action + W0
        t = [self.time[self.i], self.time[self.i+1]]
        y_dot = odeint(f, self.state, t, args=(w1, w2, w3, w4))[1]  
        # , Dfun=self.jac)[1]
        self.state = y_dot
        reward = self.reward(y_dot, action, self.i)
        done = self.is_done()
        if self.check_contained:
            done = self.is_contained(self._state)
        self.i += 1
        return self.state, reward, done, None
    
    def set_time(self, steps, dt):
        '''
        Set the time of the environment.

        Args
        ----
        steps: int
            Number of steps.
        dt: float
            Time step.
        '''
        self.dt = dt
        self.time_max = int(dt * steps)
        self.steps = steps
        self.time = np.linspace(0, self.time_max, self.steps + 1)
    
    def reset(self):
        '''
        Reset the environment.  

        Returns
        -------
        state: np.ndarray
            State of the system.
        '''
        self.i = 0
        self.state = self._observation_space.sample()
        return self.state    


class QuadcopterWrapper(gym.ObservationWrapper):
    '''
    Wrapper for the Quadcopter environment.
    '''
    def __init__(self, env: Env):
        super().__init__(env)
        low = self.observation(env._observation_space.low)
        high = self.observation(env._observation_space.high)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32)

    def observation(self, state: np.ndarray) -> np.ndarray:
        '''
        Transform the state to the observation.

        Args
        ----
        state: np.ndarray
            State of the system.

        Returns
        -------
        obs: np.ndarray
            Observation of the system.
        '''
        r = angles2rotation(state[9:])
        obs = np.zeros(18)
        obs[0:9] = state[0:9]
        obs[9:18] = r
        return obs
    
    def reset(self, **kwargs):
        '''
        Reset the environment.
        '''
        return super().reset(**kwargs)
