import os
import torch as T
import torch.nn.functional as F
import numpy as np
from ReplayBuffer import ReplayBuffer
from Networks import *


class Agent(object):
    def __init__(self, input_dims, n_actions, max_action, env, alpha=0.0003, beta=0.0003, gamma=0.99,
                 max_mem_size=1000000, tau=0.005, layer1_size=256, layer2_size=256, batch_size=8, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_mem_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.scale = reward_scale

        self.actor = ActorNet(alpha, input_dims, n_actions, max_action, name='actor')
        self.critic1 = CriticNet(beta, input_dims, n_actions, name='critic1')
        self.critic2 = CriticNet(beta, input_dims, n_actions, name='critic2')
        self.value = ValueNet(beta, input_dims, name='value')
        self.target_value = ValueNet(beta, input_dims, name='target_value')

        self.update_targetnet_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        action, _ = self.actor.sample_normal(state, reparam=False)

        return action.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def update_targetnet_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()
        target_value_dict = dict(target_value_params)
        value_dict = dict(value_params)

        for name in value_dict:
            value_dict[name] = tau * value_dict[name].clone() + (1 - tau) * target_value_dict[name].clone()
        self.target_value.load_state_dict(value_dict)

    def save_models(self, n=None):
        print('...saving models...')
        self.actor.save_checkpoint(n)
        self.critic1.save_checkpoint(n)
        self.critic2.save_checkpoint(n)
        self.value.save_checkpoint(n)
        self.target_value.save_checkpoint(n)

    def load_models(self, n=None):
        print('...loading models...')
        self.actor.load_checkpoint(n)
        self.critic1.load_checkpoint(n)
        self.critic2.load_checkpoint(n)
        self.value.load_checkpoint(n)
        self.target_value.load_checkpoint(n)

    def learn(self):
        if self.memory.mem_count < self.batch_size:
            return

        b_state, b_action, b_reward, b_state_, b_terminal = self.memory.sample_buffer(self.batch_size)
        b_state = T.tensor(b_state, dtype=T.float).to(self.actor.device)
        b_action = T.tensor(b_action, dtype=T.float).to(self.actor.device)
        b_reward = T.tensor(b_reward, dtype=T.float).to(self.actor.device)
        b_state_ = T.tensor(b_state_, dtype=T.float).to(self.actor.device)
        b_terminal = T.tensor(b_terminal).to(self.actor.device)

        values = self.value(b_state).view(-1)
        values_ = self.target_value(b_state_).view(-1)
        values_[b_terminal] = 0

        actions, log_probs = self.actor.sample_normal(b_state, reparam=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1(b_state, actions)
        q2_new_policy = self.critic2(b_state, actions)
        critic_values = T.min(q1_new_policy, q2_new_policy)
        critic_values = critic_values.view(-1)

        self.value.optimizer.zero_grad()
        values_target = critic_values - log_probs
        values_loss = 0.5 * F.mse_loss(values, values_target)
        values_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(b_state, reparam=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1(b_state, actions)
        q2_new_policy = self.critic2(b_state, actions)
        critic_values = T.min(q1_new_policy, q2_new_policy)
        critic_values = critic_values.view(-1)

        actor_loss = log_probs - critic_values
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        q_hat = self.scale * b_reward + self.gamma * values_
        q1_old_policy = self.critic1(b_state, b_action).view(-1)
        q2_old_policy = self.critic2(b_state, b_action).view(-1)
        critic1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_targetnet_parameters()