#!/usr/bin/env python3
# from cmath import cos
import numpy as np
from scipy.integrate import odeint
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
from numpy.linalg import norm
from env.constants import CONSTANTS, omega_0, W0
from env.params import ENV_PARAMS
from scipy.spatial.transform import Rotation as R

G = CONSTANTS['G']
Ixx = CONSTANTS['Ixx']
Iyy = CONSTANTS['Iyy']
Izz = CONSTANTS['Izz']
B = CONSTANTS['B']
M = CONSTANTS['M']
L = CONSTANTS['L']
K = CONSTANTS['K']
K1 = eval(ENV_PARAMS['K1'])
K11 = eval(ENV_PARAMS['K11'])
K2 = eval(ENV_PARAMS['K2'])
K21 = eval(ENV_PARAMS['K21'])
K3 = eval(ENV_PARAMS['K3'])


'''
G = 9.81
Ixx, Iyy, Izz = 1, 1, 0.5  # (4.856*10**-3, 4.856*10**-3, 8.801*10**-3)
B, M, L = 1.140 * 1e-7, 1, 0.225
K = 2.980 * 1e-6  # kt
'''

W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0


def sec(x): return 1/np.cos(x)


def angles2rotation(angles, flatten=True):
    z, y, x = angles  # psi, theta, phi
    r = R.from_euler('xyz', [x, y, z], degrees=False)
    r = r.as_matrix()
    if flatten:
        r = r.flatten()
    return r


def rotation2angles(rot):
    if rot.shape[0] == 9:
        rot = rot.reshape((3, 3))
    r = R.from_matrix(rot)
    return r.as_euler('zyx', degrees=False)


def transform_x(x):
    '''
        x_t -> o_t
    '''
    r = angles2rotation(x[9:])
    z = np.zeros(18)
    z[0:9] = x[0:9]
    z[9:18] = r
    return z


def inv_transform_x(x):
    '''
        o_t -> x_t
    '''
    r = x[9:]
    angles = rotation2angles(r)
    x[9:12] = angles
    return x[:12]


def f(X, t, w1, w2, w3, w4):  # Sistema dinámico
    '''
        f calcula el vector dot_x = f(x, t, w) (sistema dinamico);

        X: es un vector de 12 posiciones;
        t: es un intervalo de tiempo [t1, t2];
        wi: es un parametro de control, i = 1, 2, 3, 4;

        regresa dot_x
    '''
    u, v, w, _, _, _, p, q, r, _, theta, phi = X
    # Ixx, Iyy, Izz = I
    W = np.array([w1, w2, w3, w4])
    du = r * v - q * w - G * np.sin(theta)
    dv = p * w - r * u - G * np.cos(theta) * np.sin(phi)
    dw = q * u - p * v + G * np.cos(phi) * np.cos(theta) - (K/M) * norm(W) ** 2
    dp = ((L * B) / Ixx) * (w4 ** 2 - w2 ** 2) - q * r * ((Izz - Iyy) / Ixx)
    dq = ((L * B) / Iyy) * (w3 ** 2 - w1 ** 2) - p * r * ((Ixx - Izz) / Iyy)
    dr = (B/Izz) * (w2 ** 2 + w4 ** 2 - w1 ** 2 - w3 ** 2)
    dpsi = (q * np.sin(phi) + r * np.cos(phi)) * (1 / np.cos(theta))
    dtheta = q * np.cos(phi) - r * np.sin(phi)
    dphi = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
    dx = u
    dy = v
    dz = w
    return du, dv, dw, dx, dy, dz, dp, dq, dr, dpsi, dtheta, dphi


def f_x(state, actions):
    '''
        jac_f calcula el jacobiano de la funcion f;

        X: es un vector de 12 posiciones;
        t: es un intervalo de tiempo [t1, t2];
        wi: es un parametro de control, i = 1, 2, 3, 4;

        regrasa la matriz J.
    '''
    u, v, w, _, _, _, p, q, r, _, theta, phi = state
    a1 = (Izz - Iyy)/Ixx
    a2 = (Ixx - Izz)/Iyy
    J = np.zeros((12, 12))

    J[0, :] = np.array(
        [0, r, -q, 0, 0, 0, 0, -w, v, 0, - G * np.cos(theta), 0])
    J[1, :] = np.array([-r, 0, p, 0, 0, 0, w, 0, -u, 0, G *
                        np.sin(theta) * np.sin(phi),
                        -G * np.cos(theta) * np.cos(phi)])
    J[2, :] = np.array([q, -p, 0, 0, 0, 0, -v, u, 0, 0, G *
                        np.sin(theta) * np.cos(phi),
                        -G * np.cos(theta) * np.sin(phi)])

    J[3, 0] = 1.
    J[4, 1] = 1.
    J[5, 2] = 1.
    J[6, 7] = -r * a1
    J[6, 8] = -q * a1
    J[7, 6] = -r * a2
    J[7, 8] = -p * a2
    # ddr = np.zeros(12) # J[8, :]

    J[9, 7: 9] = np.array([np.sin(phi), np.cos(phi) * sec(theta)])
    J[9, 10:] = np.array([r * np.cos(phi) * np.tan(theta) * sec(theta),
                          (q * np.cos(phi) - r * np.sin(phi)) * sec(theta)])

    J[10, 7: 9] = np.array([np.cos(phi), - np.sin(phi)])
    J[10, -1] = -q * np.sin(phi) - r * np.cos(phi)

    J[11, 6: 9] = np.array(
        [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)])
    J[11, 10:] = np.array([(q * np.sin(phi) + r * np.cos(phi)) * sec(theta) **
                           2, (q * np.cos(phi) - r * np.sin(phi)) * np.tan(theta)])

    return J


def jac_f(X, t, w1, w2, w3, w4):
    '''
        jac_f calcula el jacobiano de la funcion f;

        X: es un vector de 12 posiciones;
        t: es un intervalo de tiempo [t1, t2];
        wi: es un parametro de control, i = 1, 2, 3, 4;

        regrasa la matriz J.
    '''
    u, v, w, _, _, _, p, q, r, _, theta, phi = X
    a1 = (Izz - Iyy)/Ixx
    a2 = (Ixx - Izz)/Iyy
    J = np.zeros((12, 12))
    ddu = np.zeros(12)
    J[0, 1:3] = [r, -q]
    ddv = [0, r, -q, 0, 0, 0, w, 0, -u, G *
           np.sin(theta) * np.sin(phi), -G * np.cos(theta) * np.cos(phi)]
    ddw = [q, -p, 0, 0, 0, 0, -v, u, 0, 0, G *
           np.sin(theta) * np.cos(phi), -G * np.cos(theta) * np.sin(phi)]
    ddx = np.zeros(12)
    ddx[0] = 1
    ddy = np.zeros(12)
    ddx[1] = 1
    ddz = np.zeros(12)
    ddx[2] = 1
    ddp = np.zeros(12)
    ddp[7] = -r * a1
    ddp[8] = -q * a1
    ddq = np.zeros(12)
    ddq[6] = -r * a2
    ddq[8] = -p * a2
    ddr = np.zeros(12)

    ddpsi = np.zeros(12)
    ddpsi[7: 9] = [np.sin(phi), np.cos(phi) * sec(theta)]
    ddpsi[10:] = [r * np.cos(phi) * np.tan(theta) * sec(theta),
                  (q * np.cos(phi) - r * np.sin(phi)) * sec(theta)]

    ddtheta = np.zeros(12)
    ddtheta[7: 9] = [np.cos(phi), - np.sin(phi)]
    ddtheta[-1] = -q * np.sin(phi) - r * np.cos(phi)

    ddphi = np.zeros(12)
    ddphi[6: 9] = [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)]
    ddphi[10:] = [(q * np.sin(phi) + r * np.cos(phi)) * sec(theta) **
                  2, (q * np.cos(phi) - r * np.sin(phi)) * np.tan(theta)]

    J = np.array([ddu, ddv, ddw, ddx, ddy, ddz, ddp,
                 ddq, ddr, ddpsi, ddtheta, ddphi])
    return J


def f_u(state, actions):
    w1, w2, w3, w4 = actions
    J = np.zeros((12, 4))
    J[2, :] = -2 * (K/M) * actions
    J[3, :] = 2 * (L/Ixx) * np.array([0, -w2, 0, w4])
    J[4, :] = 2 * (L/Iyy) * np.array([-w1, 0, w3, 0])
    J[5, :] = 2 * (B/Izz) * actions
    return J


def penalty(state, action, i=None):
    '''
    falta
    '''
    # u, v, w, x, y, z, p, q, r, psi, theta, phi = state
    return terminal_penalty(state, i) + K3 * norm(action)


def terminal_penalty(state, i=None):
    # u, v, w, x, y, z, p, q, r, psi, theta, phi = state
    penalty = K1 * norm(state[3:6])
    penalty += K11 * norm(state[:3])
    mat = angles2rotation(state[9:], flatten=False)
    penalty += K21 * norm(state[6:9])
    penalty += K2 * norm(np.identity(3) - mat)
    return penalty
