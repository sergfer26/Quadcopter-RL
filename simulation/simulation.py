import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def rollout(agent, env, flag=False, state_init=None):
    '''
    Simulación de interacción entorno-agente

    Argumentos
    ----------
    agent : `(DDPG.DDPGAgent, Linear.Agent, GPS.iLQRAgent, models.ActorCriticDDPG, models.ActorCriticTD3)`
        Instancia que representa al agente que toma acciones 
        en la simulación.
    env : `gym.Env`
        Entorno de simualción de gym.
    flag : bool
        ...
    state_init : `np.ndarray`
        Arreglo que representa el estado inicial de la simulación.

    Retornos
    --------
    states : `np.ndarray`
        Trayectoria de estados en la simulación con dimensión (env.steps, n_x).
    acciones : `np.ndarray`
        Trayectoria de acciones en la simulación con dimensión 
        (env.steps -1, n_u).
    scores : `np.ndarray`
        Trayectoria de puntajes (incluye reward) en la simulación con
        dimensión (env.steps -1, ?).
    '''
    # t = env.time
    env.flag = flag
    state = env.reset()
    if hasattr(agent, 'reset') and callable(getattr(agent, 'reset')):
        agent.reset()
    if isinstance(state_init, np.ndarray):
        env.state = state_init
        state = state_init
    states = np.zeros((env.steps + 1, env.observation_space.shape[0]))
    actions = np.zeros((env.steps, env.action_space.shape[0]))
    scores = np.zeros((env.steps, 2))  # r_t, Cr_t
    states[0] = state
    episode_reward = 0
    i = 0
    while True:
        action = agent.get_action(state)
        new_state, reward, done, info = env.step(action)
        episode_reward += reward
        states[i + 1] = state
        if isinstance(info, dict) and ('real_action' in info.keys()):
            action = info['real_action']  # env.action(action)
        actions[i] = action
        scores[i] = np.array([reward, episode_reward])
        state = new_state
        if done:
            break
        i += 1
    return states, actions, scores


def n_rollouts(agent, env, n, flag=False, states_init=None,
               t_x=None, t_u=None):
    '''
    Retornos
    --------
    n_states: `np.ndarray`
        Colección de `n` trayectoria de estados en la simulación con dimensión 
        `(n, env.steps, n_x)`.
    n_actions: 
        Colección de `n` trayectoria de acciones en la simulación con dimensión 
        `(n, env.steps -1, n_u)`.
    n_scores:
        Colección de `n` trayectoria de puntajes (incluye reward) en la simulación con
        dimensión `(n, env.steps -1, ?)`.
    '''
    n_states = np.zeros((n, env.steps + 1, env.observation_space.shape[0]))
    n_actions = np.zeros((n, env.steps, env.action_space.shape[0]))
    n_scores = np.zeros((n, env.steps, 2))
    state_init = None
    for k in range(n):  # for k in progressbar(range(n)):
        if isinstance(states_init, np.ndarray):
            if len(states_init.shape) == 2:
                state_init = states_init[k, :]
            else:
                state_init = states_init
        states, actions, scores = rollout(
            agent, env, flag=flag, state_init=state_init)
        n_states[k] = states
        n_actions[k] = actions
        n_scores[k] = scores
    if callable(t_x):
        n_states = np.apply_along_axis(t_x, -1, n_states)
    if callable(t_u):
        n_actions = np.apply_along_axis(t_u, -1, n_actions)
    return n_states, n_actions, n_scores


def plot_rollouts(array: np.ndarray, time: np.ndarray, columns: list,
                  axes=None, subplots=True, dpi=150, colors=None, alpha=0.4,
                  ylims=None, style="fivethirtyeight"):
    '''
    array : `np.ndarray`
        ...
    time : `np.ndarray`
        ...
    columns : `list`
        ...
    ax : ...
        ...
    dpi : `int`
        ...
    ylims : `np.ndarray`
        Valores limites de los ejes `ax`. Si `subplots=True` (n_x, 2), 
        en otro caso (2,).
    '''
    plt.style.use(style)
    if len(array.shape) == 2:
        array = array.reshape(1, array.shape[0], array.shape[1])
    samples, steps, n_var = array.shape
    if not isinstance(colors, list):
        # Use seaborn's "Set1" color palette
        colors = plt.cm.jet(np.linspace(0, 1, len(columns))) 
    if len(colors) == 1:
        colors *= array.shape[-1]


    fig = None
    if not isinstance(axes, np.ndarray) and not isinstance(axes, plt.Axes):
        if subplots:
            fig, axes = plt.subplots(n_var // 2, 2, dpi=dpi, sharex=True)
        else:
            fig, axes = plt.subplots(dpi=dpi)
    for k in range(samples):

        # data = pd.DataFrame(array[k, :, :], columns=columns)
        data = array[k, :, :]
        # data['$t (s)$'] = time[0: steps]
        t = time[:steps]
        if k == 0:
            legend = True
        else:
            legend = False

        # data.plot(x='$t (s)$', subplots=subplots,
        #           ax=ax, legend=legend, alpha=alpha,
        #           colormap=cmap
        #           )
        if subplots:
            for i, (ax, col) in enumerate(zip(axes.flatten(), columns)):
                ax.plot(t, data[:, i], label=col, alpha=alpha, color=colors[i])
                if legend:
                    ax.legend()
        else:
            for i, col in enumerate(columns):
                axes.plot(t, data[:, i], label=col, alpha=alpha, color=colors[i])
                if legend:
                    axes.legend()

    if isinstance(ylims, np.ndarray):
        if subplots:
            for e, ax in enumerate(axes.flatten()):
                ax.set_ylim(ymin=ylims[e, 1], ymax=ylims[e, 0])
        else:
            axes.set_ylim(ymin=ylims[0], ymax=ylims[1])

    if not pd.isna(fig):
        fig.set_size_inches(18.5, 10.5)
    return fig, axes