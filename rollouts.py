import os
import time 
import torch
import pathlib
import argparse
import numpy as np
import matplotlib as mpl
import multiprocessing as mp

from gym import spaces
from typing import Union
from loguru import logger
from multiprocessing import Process
from matplotlib import pyplot as plt

from utils import send_resport
from simulation import n_rollouts

from env.params import STATE_NAMES
from env import QuadcopterEnv, QuadcopterWrapper
from env.params import state_space as STATE_SPACE
from env.equations import (
    inv_transform_x,
    transform_x
)


def plot_classifier(
        states, 
        cluster, 
        x_label: str = 'x', 
        y_label: str = 'y',
        figsize=(6, 6), 
        dpi: int = 300, 
        ax = None,
        style: str = "fivethirtyeight"
        ):
    cmap = None
    plt.style.use(style)
    if not isinstance(ax, plt.Axes):
        ax = plt.subplots(figsize=figsize, dpi=dpi)[1]
    if cluster.all():
        cluster = 'blue'
    else:
        cmap = mpl.colors.ListedColormap(['red', 'blue'])
    sc = ax.scatter(states[0], states[1], c=cluster, s=10, alpha=0.3,
                    cmap=cmap)

    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    return ax, sc


def rollout4mp(agent, env, mp_list, n=1, states_init=None, inv_transform_x=None):
    '''
    states : (n, env.steps, env.observation_space.shape[0])
    '''
    states = n_rollouts(agent, env, n=n, t_x=inv_transform_x, states_init=states_init)[0]
    mp_list.append(states)


def rollouts(agent, env, sims, state_space, num_workers=None,
             inv_transform_x=None, transform_x=None):
    '''
    Retorno
    --------
    states : (np.ndarray)
        dimensión -> (state_space.shape[0], sims, env.steps,
                        env.observation_space.shape[0])
    '''
    if not isinstance(num_workers, int):
        num_workers = state_space.shape[1]

    states = mp.Manager().list()
    process_list = list()
    if hasattr(agent, 'env'):
        other_env = agent.env
    else:
        other_env = env
    init_states = np.empty((num_workers, sims, env.state.shape[0]))
    for i in range(num_workers):
        env.observation_space = spaces.Box(
            low=state_space[0, i], high=state_space[1, i], dtype=np.float64)
        init_states[i] = np.array(
            [env.observation_space.sample() for _ in range(sims)])
        init_state = init_states[i]
        if callable(transform_x):
            init_state = np.apply_along_axis(transform_x, -1, init_state)
        p = Process(target=rollout4mp, args=(
            agent, other_env, states, sims, init_state, inv_transform_x
        )
        )
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()

    states = np.array(list(states))
    # if callable(inv_transform_x):
    #     states = np.apply_along_axis(inv_transform_x, -1, states)

    return states


def classifier(state: np.ndarray, c: float = 5e-1, mask: np.ndarray = None,
               ord: Union[int, str] = 2) -> np.ndarray:
    '''
    ord : {int, str}
    '''
    ord = int(ord) if ord.isdigit() else np.inf
    if isinstance(mask, np.ndarray):
        state = state[mask]
    return np.linalg.norm(state, ord=ord) < c


def confidence_region(states: np.ndarray, c: float = 5e-1, mask: np.ndarray = None,
                      ord: Union[int, float, str] = 2) -> np.ndarray:
    '''
    ord : {int, str: inf}
    '''
    return np.apply_along_axis(classifier, -1, states, c, mask, ord)


def get_color(bools):
    return np.array(['b' if b else 'r' for b in bools])

def main(args):
    save_path = '/'.join(args.policy_path.split('/')[:-1])
    logger.info(f"[*] Save path at {save_path}")

    sims = args.sims

    labels = [('$u$', '$x$'), ('$v$', '$y$'), ('$w$', '$z$'),
              ('$p$', '$\phi$'), ('$q$', '$\\theta$'),
              ('$r$', '$\psi$')
              ]
    env = QuadcopterEnv()
    T = int(60 / env.dt)
    list_steps = np.array([5, 15, 30, 60]) / env.dt
    if not os.path.exists(f'{save_path}/states_60.npz'):
        # 1. Setup
        policy = torch.load(args.policy_path)

        env = QuadcopterWrapper(QuadcopterEnv())
        env.set_time(T, env.dt)
        # 3. Policy's simulations
        logger.info(f"Policy's {sims} simulations started.")

        start_time = time.time()
        states = rollouts(policy, env, sims, STATE_SPACE,
                          inv_transform_x=None,
                          transform_x=transform_x)
        end_time = time.time()
        logger.info(f"Policy's simulations ended, time: {end_time -start_time}.")
        mask1 = np.apply_along_axis(lambda x, y: np.greater(
            abs(x), y), -1, states[:, 0, 0], 0)
        mask2 = STATE_SPACE[1] > 0
        indices = np.array([np.where(np.all(mask1 == mask2[i], axis=1))[0]
                            for i in range(6)]).squeeze()
        states = states[indices]
        array_path = save_path + f'states_{int(env.time_max)}.npz'
        np.savez(
            array_path,
            states=states,
            high=STATE_SPACE[1]
        )
        logger.info(f"[*] Simulations saved at {array_path}")
    else:
        states = np.load(f'{save_path}/states_60.npz')['states']

    init_states = states[:, :, 0]
    # steps=int(t * env.dt))
    state_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    for t in list_steps:
        bool_state = confidence_region(
            states[:, :, int(t)],
            c=args.threshold,
            mask=state_mask,
            ord=args.ord
        )
        # cluster = np.apply_along_axis(get_color, -1, bool_state)
        fig, axes = plt.subplots(figsize=(14, 10), nrows=len(labels)//3,
                                 ncols=3, dpi=250, sharey=False)
        axs = axes.flatten()
        for i in range(init_states.shape[0]):
            mask = abs(init_states[i, 0]) > 0
            label = np.array(STATE_NAMES)[mask]
            plot_classifier(
                init_states[i, :, mask],
                bool_state[i], x_label=label[0],
                y_label=label[1],
                ax=axs[i])
        fig.suptitle(f'Política, tiempo: {t * env.dt}')
        image_path = save_path + f'samples_policy_{int(t * env.dt)}.png'
        logger.info(f"[*] Image saved at {image_path}")
        fig.savefig(image_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy-path', type=str,
                        default='results/2025-10-10_ddpg/2025-10-10_19-30-53-ddpg_s0/pyt_save/policy')
    parser.add_argument('--sims', type=int, default=int(1e4))
    parser.add_argument('--send-mail', action='store_true',
                        default=False, help='Enable sending mail')
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--ord', default='inf', type=str)
    args = parser.parse_args()

    if args.send_mail:
        send_resport(main, args=[args], subject="Estabilización DDPG")
    else:
        main(args)