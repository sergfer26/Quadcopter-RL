import os 
import numpy as np
import torch
from torch.autograd import Variable
import spinup.algos.pytorch.td3.core as td3
from torch.nn.modules import ReLU
from .mlp import Actor, Critic


class ActorCritic(td3.MLPActorCritic):

    def __init__(self, observation_space, action_space, hidden_sizes=[128, 128], activation=ReLU, checkpoint: str =None):
        super().__init__(observation_space, action_space, hidden_sizes, activation)
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = Actor(obs_dim, act_dim, act_limit, hidden_sizes=hidden_sizes) # MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = Critic(obs_dim, act_dim, hidden_sizes=hidden_sizes)
        self.q2 = Critic(obs_dim, act_dim, hidden_sizes=hidden_sizes)

        if checkpoint:
            if os.path.exists(checkpoint) :
                self.pi.load(path=checkpoint)

    def get_action(self, state: np.ndarray):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        return self.act(state)[0]