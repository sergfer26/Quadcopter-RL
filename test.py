from env import QuadcopterEnv, QuadcopterWrapper
from models import ActorCriticDDPG
from matplotlib import pyplot as plt
from env.equations import inv_transform_x
from simulation import plot_rollouts, n_rollouts
from env.params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES, ENV_PARAMS, STATE_PARAMS

env = QuadcopterWrapper(QuadcopterEnv())
path = 'saved_policies/best_gps/policy'
ac_kwargs = dict(hidden_sizes=[128] * 2, checkpoint=path)
agent = ActorCriticDDPG(env.observation_space, env.action_space, **ac_kwargs)
n= 100
states, actions, scores = n_rollouts(agent, env, n, t_x=inv_transform_x)
fig1, _ = plot_rollouts(states, env.time, STATE_NAMES, alpha=0.05)
fig1.savefig('state_rollouts.png')
fig2, _ = plot_rollouts(actions, env.time, ACTION_NAMES, alpha=0.05)
fig2.savefig('action_rollouts.png')
fig3, _ = plot_rollouts(scores, env.time, REWARD_NAMES, alpha=0.05)
fig3.savefig('score_rollouts.png')


