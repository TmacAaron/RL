import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class CriticNet(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir='tmp/sac'):
        super(CriticNet, self).__init__()
        self.beta = beta
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, self.name+'_sac')

        self.fc1 = nn.Linear(self.input_dims[0]+self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = T.device('cuda:0' if T.cuda.is_available else 'cpu')
        self.to(self.device)
        print(self.device)

    def forward(self, state, action):
        state_value = self.fc1(T.cat([state, action], dim=1))
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        q = self.q(state_value)

        return q

    def save_checkpoint(self, n=None):
        if n is None:
            T.save(self.state_dict(), self.checkpoint_file)
        else:
            T.save(self.state_dict(), os.path.join(self.chkpt_dir, self.name+'_'+n+'_sac'))

    def load_checkpoint(self, n=None):
        if n is None:
            self.load_state_dict(T.load(self.checkpoint_file))
        else:
            self.load_state_dict(T.load(os.path.join(self.chkpt_dir, self.name+'_'+n+'_sac')))


class ValueNet(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256, name='value', chkpt_dir='tmp/sac'):
        super(ValueNet, self).__init__()
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, self.name + '_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        v = self.v(state_value)

        return v

    def save_checkpoint(self, n=None):
        if n is None:
            T.save(self.state_dict(), self.checkpoint_file)
        else:
            T.save(self.state_dict(), os.path.join(self.chkpt_dir, self.name + '_' + n + '_sac'))

    def load_checkpoint(self, n=None):
        if n is None:
            self.load_state_dict(T.load(self.checkpoint_file))
        else:
            self.load_state_dict(T.load(os.path.join(self.chkpt_dir, self.name + '_' + n + '_sac')))


class ActorNet(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, max_action, fc1_dims=256, fc2_dims=256, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNet, self).__init__()
        self.alpha = alpha
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.max_action = max_action
        self.reparam_noise = 1e-6
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, self.name + '_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparam=True):
        mu, sigma = self.forward(state)
        probs = Normal(mu, sigma)

        if reparam:
            actions = probs.rsample()
        else:
            actions = probs.sample()

        action = T.tanh(actions).to(self.device)
        log_probs = probs.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = T.sum(log_probs, dim=1, keepdim=True)

        return action*T.tensor(self.max_action).to(self.device), log_probs

    def save_checkpoint(self, n=None):
        if n is None:
            T.save(self.state_dict(), self.checkpoint_file)
        else:
            T.save(self.state_dict(), os.path.join(self.chkpt_dir, self.name + '_' + n + '_sac'))

    def load_checkpoint(self, n=None):
        if n is None:
            self.load_state_dict(T.load(self.checkpoint_file))
        else:
            self.load_state_dict(T.load(os.path.join(self.chkpt_dir, self.name + '_' + n + '_sac')))
