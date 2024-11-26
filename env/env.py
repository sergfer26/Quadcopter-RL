import gym
import numpy as np
from copy import copy
from gym.core import Env
from .equations import f
from .constants import W0, omega_0, C
from .equations import angles2rotation, penalty
from scipy.integrate import odeint
from .params import ENV_PARAMS, STATE_PARAMS

omega0_per = .60
HIGH_ACT = omega_0 * omega0_per  # 60 #Velocidad maxima de los motores 150
LOW_ACT = - omega_0 * omega0_per
LOW_OBS = np.array([- eval(v) for v in STATE_PARAMS.values()])
HIGH_OBS = np.array([eval(v) for v in STATE_PARAMS.values()])
DT = ENV_PARAMS['dt']
STEPS = ENV_PARAMS['STEPS']


def get_reward(x, u, i):
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
                 high_act: np.ndarray = HIGH_ACT * np.ones(4)) -> None:
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

    def is_contained(self, state):
        '''
            Verifica si los valores de (x, y, z) estan dentro
            de los rangos definidos por el ambiente.
            Params:
                * state (array): vector de 12 o 18 posiciones.
            Salidas:
                * (bool): estado de contención.
        '''
        x = state[3:6]
        high = self._observation_space.high[3:6]
        low = self._observation_space.low[3:6]
        aux = np.logical_and(low <= x, x <= high)
        return aux.all()

    def is_done(self):
        '''
            is_done verifica si el drone ya termino de hacer su tarea;

            regresa valor booleano.
        '''
        done = False
        if self.i == self.steps-1:  # Si se te acabo el tiempo
            done = True
        return done
        
    def step(self, action):
        '''
        Realiza la interaccion entre el agente y el ambiente en
        un paso de tiempo.

        Argumentos
        ----------
        action: `np.ndarray`
            Representa la acción actual ($a_t$). Arreglo de dimesnión (4,) 
            con valores entre [low, high].

        Retornos
        --------
        state : `np.ndarray`
            Representa el estado siguiente ($s_{t+1}$). Arreglo de dimensión (12,).
        reward : `float`
            Representa la recompensa siguiente ($r_{t+1}$).
        done : `bool`
            Determina si la trayectoria ha terminado o no.
        info : `dict`
            Es la información  adicional que proporciona el sistema. 
            info['real_action'] : action.
        '''
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
        set_time fija la cantidad de pasos y el tiempo de simulación;
        steps: candidad de pasos (int);
        time_max: tiempo de simulación (float).
        '''
        self.dt = dt
        self.time_max = int(dt * steps)
        self.steps = steps
        self.time = np.linspace(0, self.time_max, self.steps + 1)
    
    def reset(self):
        '''
            reset fija la condición inicial para cada simulación del drone
            regresa el estado actual del drone.
        '''
        self.i = 0
        self.state = self._observation_space.sample()
        return self.state    


class QuadcopterWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        low = self.observation(env._observation_space.low)
        high = self.observation(env._observation_space.high)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32)

    def observation(self, state):
        '''
            x_t -> o_t
        '''
        r = angles2rotation(state[9:])
        obs = np.zeros(18)
        obs[0:9] = state[0:9]
        obs[9:18] = r
        return obs
    
    def reset(self, **kwargs):
        return super().reset(**kwargs)
    

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, 
                 mu=0.0,
                 theta=0.15,
                 max_sigma=0.9,
                 min_sigma=0.05,
                 decay_period=1e4
                 ):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.action_dim)
        self.state = x + dx
        return np.clip(self.state, self.low, self.high)

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - \
            (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)