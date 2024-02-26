import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[128, 128]):
        super(Critic, self).__init__()
        # Definición de la arquitectura
        h_sizes = hidden_sizes.copy()
        h_sizes.insert(0, state_dim)
        h_sizes.insert(0, state_dim + action_dim)

        # Definición de las capas ocultas
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))

        self.out = nn.Linear(h_sizes[-1], 1)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.out(x)

        return x
    
    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device))


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, act_limit, hidden_sizes=[128, 128]):
        super(Actor, self).__init__()
        # Definición de la arquitectura
        h_sizes = hidden_sizes.copy()
        h_sizes.insert(0, state_dim)
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))

        self.out = nn.Linear(h_sizes[-1], action_dim)
        self.act_limit = act_limit

    def forward(self, state):
        """
        Param state is a torch tensor
        """

        x = state
        # import pdb; pdb.set_trace()
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = torch.tanh(self.out(x))
        return self.act_limit * x
    
    def load(self, path):
        self.load_state_dict(torch.load(path))