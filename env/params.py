
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


STATE_PARAMS = {'$u$': '1.0', '$v$': '1.0', '$w$': '1.0',
              '$x$': '2.0', '$y$': '2.0', '$z$': '2.0',
              '$p$': '0.5', '$q$': '0.5', '$r$': '0.5',
              '$\psi$': 'np.pi/64', r'$\theta$': 'np.pi/64',
              '$\\varphi$': 'np.pi/64'}


ENV_PARAMS = {'dt': 0.04, 'STEPS': 750, 'omega0_per': 0.60,
              'K1': '10', 
              'K11': '10', 
              'K2': '100', 
              'K21': '10', 
              'K3': '.5',
              'K12': '1.0',
              'K22': '1.3'
              }
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
