
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


STATE_PARAMS = {'$u$': '0.0', '$v$': '0.0', '$w$': '0.0',
              '$x$': '8', '$y$': '8', '$z$': '8',
              '$p$': '0.00', '$q$': '0.0', '$r$': '0.0',
              '$\psi$': 'np.pi/32', r'$\theta$': 'np.pi/32',
              '$\\varphi$': 'np.pi/32'}


ENV_PARAMS = {'dt': 0.04, 'STEPS': 750, 'omega0_per': 0.60,
              'K1': '10', 'K11': '10', 'K2': '100', 'K21': '10', 'K3': '.5'}
# Si es false los vuelos pueden terminar


# Etiquetas
STATE_NAMES = list(STATE_PARAMS.keys())

ACTION_NAMES = [f'$\\omega_{i}$' for i in range(1, 5)]

REWARD_NAMES = ['$r_t$', r'$\sum r_t$']

high = np.array([
    # u, v, w, x, y, z, p, q, r, psi, theta, phi
    [10., 0., 0., 20., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 10., 0., 0., 20., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 12., 0., 0., 25., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., .0, 0., 0., 0., np.pi/2],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., np.pi/2, 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1., np.pi/2, 0., 0.]
])

low = -high
state_space = np.stack([low, high])


# N es el numero de vuelos hechos con el control lineal
# n es el numéro de vuelos de simulación de la red neuronal

# Define colors for the colormap
colors = ['royalblue', 'royalblue',
          'mediumpurple', 'mediumpurple',
          'orchid', 'orchid',
          'royalblue', 'royalblue',
          'mediumpurple', 'mediumpurple',
          'orchid', 'orchid'
          ]

# Create a custom colormap
state_cmap = LinearSegmentedColormap.from_list('CustomColormap', colors, N=12)

# Set ylimists for state variables
