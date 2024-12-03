import json
import torch
import argparse
import pathlib
import numpy as np
import pandas as pd 

from matplotlib import pyplot as  plt
from env.equations import inv_transform_x
from spinup import ddpg_pytorch, td3_pytorch
from utils import send_resport, create_report
from models import ActorCriticDDPG, ActorCriticTD3
from env import QuadcopterEnv, QuadcopterWrapper
from spinup.utils.run_utils import setup_logger_kwargs
from simulation import plot_rollouts, create_animation, n_rollouts
from env.params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES, ENV_PARAMS, STATE_PARAMS
from env.noise import OUNoise




def plot_returns(data: pd.DataFrame, mean_key: str, std_key: str, ax=None):
    '''
    Plot the mean and std of the returns
    
    Args
    ----
    data: pd.DataFrame
        Dataframe with the data to plot
    mean_key: str
        Key of the mean returns
    std_key: str
        Key of the std of the returns
    ax: plt.Axes
        Axes to plot the data on
     
    Returns
    -------
    fig: plt.Figure
        Figure with the plot
    ax: plt.Axes
        Axes with the plot
    '''
    mean = data[mean_key].to_numpy()
    std = data[std_key].to_numpy()
    epochs = data["Epoch"].to_numpy()
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(epochs, mean, label=mean_key)
    ax.fill_between(epochs, mean-std, mean + std ,alpha=0.3)
    ax.set_xlabel('epochs')
    ax.set_ylabel('returns')
    ax.legend()
    return fig, ax


def main(args):
     args_json = json.dumps(vars(args), indent=4)
     print("Training Arguments: ")
     print(args_json)
     print("-"*50)

     logger_kwargs = setup_logger_kwargs(args.method, args.seed, 
                                         data_dir='results', datestamp=True)
     output_dir = logger_kwargs["output_dir"]

     noise = None
     if args.noise_ou:
          noise = OUNoise(env.action_space, args.noise_mu, args.noise_theta, args.noise_max_sigma, args.noise_min_sigma, args.noise_decay_period)
     env_fn = lambda : QuadcopterWrapper(QuadcopterEnv(noise=noise))
     if args.checkpoint: 
          path = '/home/miguel.fernandez/Quadcopter-Deep-RL/results_gps/24_10_22_09_57/policy' # 'saved_policies/best_gps/policy'
          print(f'loading model froom path: {path}')
     else:
          path = None

     ac_kwargs = dict(hidden_sizes=[args.hid]*args.l, checkpoint=path)
     kwargs = dict(env_fn=env_fn, 
                    actor_critic=None,
                    ac_kwargs=ac_kwargs, 
                    seed=args.seed, 
                    epochs=args.epochs,
                    replay_size=args.replay_size,
                    logger_kwargs=logger_kwargs, 
                    steps_per_epoch=args.steps_per_epoch,
                    gamma=args.gamma,
                    polyak=args.polyak,
                    pi_lr=args.pi_lr,
                    q_lr=args.q_lr,
                    batch_size=args.batch_size,
                    start_steps=args.start_steps,
                    update_after=args.update_after,
                    update_every=args.update_every,
                    act_noise=args.act_noise,
                    num_test_episodes=args.num_val_episodes,
                    max_ep_len=ENV_PARAMS["STEPS"],
                    save_freq=args.save_freq
     ) 
     # 1. Fitting
     if args.method == 'ddpg':
          kwargs['actor_critic'] = ActorCriticDDPG
          ddpg_pytorch(**kwargs)

     elif args.method == 'td3':
          kwargs['actor_critic'] = ActorCriticTD3
          kwargs['target_noise'] = args.target_noise
          kwargs['noise_clip'] = args.noise_clip
          kwargs['policy_delay'] = args.policy_delay
          td3_pytorch(**kwargs)

     # 2. Testing
          
     # 2.1 Setup enviorement and agent
     env = QuadcopterWrapper(QuadcopterEnv())
     # ac_kwargs["checkpoint"] = f"{output_dir}/pyt_save/model.pt"
     # if args.method == 'ddpg':
     #      agent = ActorCriticDDPG(env.observation_space, env.action_space, **ac_kwargs)
     # elif args.method == 'td3':
     #      agent = ActorCriticTD3(env.observation_space, env.action_space, **ac_kwargs)
     agent = torch.load(f"{output_dir}/pyt_save/model.pt")

     # 2.2 Simulation
     n = args.num_test_episodes
     # u, v, w, x, y, z, p, q, r, psi, theta, phi
     states, actions, scores = n_rollouts(agent, env, n, t_x=inv_transform_x)
     print('Termino de simualción...')

     # 2.3 Plot 2D simulations
     fig1, _ = plot_rollouts(states, env.time, STATE_NAMES, alpha=0.05)
     fig1.savefig(f'{output_dir}/state_rollouts.png')
     fig2, _ = plot_rollouts(actions, env.time, ACTION_NAMES, alpha=0.05)
     fig2.savefig(f'{output_dir}/action_rollouts.png')
     fig3, _ = plot_rollouts(scores, env.time, REWARD_NAMES, alpha=0.05)
     fig3.savefig(f'{output_dir}/score_rollouts.png')

     # 2.4 Plot 3D simulations
     subpath = f'{output_dir}/sample_rollouts/'
     pathlib.Path(subpath).mkdir(parents=True, exist_ok=True)
     sample_indices = np.random.randint(states.shape[0], size=args.num_anim_episodes)
     states_samples = states[sample_indices]
     actions_samples = actions[sample_indices]
     scores_samples = scores[sample_indices]

     path = f"{output_dir}/progress.txt"
     data = pd.read_csv(path, delimiter='\t', encoding='utf-8')
     fig, ax = plt.subplots(1, 2, dpi=150)
     for e, keys in enumerate([('AverageEpRet', 'StdEpRet'), ('AverageTestEpRet', 'StdTestEpRet')]):
          mean_key, std_key = keys
          plot_returns(data, mean_key, std_key, ax=ax[e])
     fig.savefig(f'{output_dir}/train_performance.png')

     create_report(output_dir, method=args.method, state_params=STATE_PARAMS, env_params=ENV_PARAMS)
     
     print('Rendering animation...')
     create_animation(states_samples,
                      actions_samples, 
                      env.time,
                      scores=scores_samples,
                      state_labels=STATE_NAMES,
                      action_labels=ACTION_NAMES,
                      score_labels=REWARD_NAMES,
                      path=subpath
                      )
     return output_dir

     # 2.5 Report results




if __name__ == '__main__':

     parser = argparse.ArgumentParser()

     # Training arguments 
     parser.add_argument('--hid', type=int, default=128)
     parser.add_argument('--l', type=int, default=2)
     parser.add_argument('--gamma', type=float, default=0.99)
     parser.add_argument('--seed', '-s', type=int, default=0)
     parser.add_argument('--epochs', type=int, default=5)
     parser.add_argument('--replay-size', type=int, default=int(1e5))
     parser.add_argument('--polyak', type=float, default=1e-2) # -> rho = \tau -1 (multiplies target)
     parser.add_argument('--steps-per-epoch', type=int, default=800 * 750)
     parser.add_argument('--pi-lr', type=float, default=1e-3)
     parser.add_argument('--q-lr', type=float, default=1e-3)
     parser.add_argument('--batch-size', type=int, default=1024)
     parser.add_argument('--start-steps', type=int, default=0)
     parser.add_argument('--update-after',  type=int, default=1000)
     parser.add_argument('--update-every', type=int, default=50)
     parser.add_argument('--act-noise', type=float, default=0.1)
     parser.add_argument('--num-val-episodes', type=int, default=100)
     parser.add_argument('--num-test-episodes', type=int, default=1000)
     parser.add_argument('--num-anim-episodes', type=int, default=3)
     parser.add_argument('--save_freq', type=int, default=1)

     # Saving and reporting arguments
     parser.add_argument('--method', type=str, default='ddpg')
     parser.add_argument('--checkpoint', action='store_true', default=True, help='Enable loading policy checkpoint')
     parser.add_argument('--no-checkpoint', dest='checkpoint', action='store_false', help="Disable loading policy checkpoint")
     parser.add_argument('--send-mail', action='store_true', default=True, help='Enable sending mail')
     parser.add_argument('--no-mail', dest='send_mail', action='store_false', help="Disable sending mail")
     parser.add_argument('--check-contained', action='store_true', default=False, help='Checks if drone contained during sim')
     parser.add_argument('--mail-subject', default=None, type=str)

     # TD3 specific arguments
     parser.add_argument('--target-noise', type=float, default=0.2)
     parser.add_argument('--noise-clip', type=float, default=0.5)
     parser.add_argument('--policy-delay', type=int, default=2)

     # Noise arguments
     parser.add_argument('--noise-ou', action='store_true', default=False, help='Enable OU noise')
     parser.add_argument('--noise-mu', type=float, default=0.0)
     parser.add_argument('--noise-theta', type=float, default=0.15)
     parser.add_argument('--noise-max-sigma', type=float, default=0.9)
     parser.add_argument('--noise-min-sigma', type=float, default=0.05)
     parser.add_argument('--noise-decay-period', type=int, default=1e4)

     args = parser.parse_args()
     if args.send_mail:
          subject = f'Entrenamiento {args.method}'
          if isinstance(args.mail_subject, str):
               subject = args.mail_subject
          send_resport(main, args=[args], subject=subject)
     else:
          main(args)





