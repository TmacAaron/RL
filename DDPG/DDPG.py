import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
    def __init__(self, max_mem_size, input_shape, n_actions):
        self.mem_size = max_mem_size
        self.mem_count = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.next_state_memory = np.zeros((self.mem_size, *input_shape))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_count % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = 1 - done
        self.mem_count += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_count, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminals


class CriticNet(nn.Module):
    def __init__(self, lr_c, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super(CriticNet, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        torch.nn.init.uniform_(self.q.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_c)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)
        print(f'set critic_net to {self.device}')

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = self.action_value(action)
        action_value = F.relu(action_value)

        state_action_value = torch.add(state_value, action_value)
        state_action_value = self.q(F.relu(state_action_value))

        return state_action_value
    
    def save_checkpoint(self):
        print(f'...save checkpoint in {self.checkpoint_file}...')
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('...load checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNet(nn.Module):
    def __init__(self, lr_a, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super(ActorNet, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        torch.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_a)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)
        print(f'set actor_net to {self.device}')

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.tanh(self.mu(x))

        return x

    def save_checkpoint(self):
        print(f'...save checkpoint in {self.checkpoint_file}...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...load checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent(object):
    def __init__(self, lr_c, lr_a, input_dims, tau, env, gamma=0.99, n_actions=2,
                 max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNet(lr_a, input_dims, layer1_size, layer2_size, n_actions, name='Actor')
        self.target_actor = ActorNet(lr_a, input_dims, layer1_size, layer2_size, n_actions, name='TargetActor')
        self.critic = CriticNet(lr_c, input_dims, layer1_size, layer2_size, n_actions, name='Critic')
        self.target_critic = CriticNet(lr_c, input_dims, layer1_size, layer2_size, n_actions, name='TargetCritic')

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_target_net_parameters(tau=1)

    def choose_action(self, observation, evaluate=False):
        self.actor.eval()
        observation = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
        mu = self.actor(observation).to(self.actor.device)
        if evaluate:
            mu_prime = mu
        else:
            mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).to(self.actor.device)
            self.actor.train()

        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if self.memory.mem_count < self.batch_size:
            return
        b_state, b_action, b_reward, b_next_state, b_terminal = self.memory.sample_buffer(self.batch_size)
        b_state = torch.tensor(b_state, dtype=torch.float).to(self.critic.device)
        b_action = torch.tensor(b_action, dtype=torch.float).to(self.critic.device)
        b_reward = torch.tensor(b_reward, dtype=torch.float).to(self.critic.device)
        b_next_state = torch.tensor(b_next_state, dtype=torch.float).to(self.critic.device)
        b_terminal = torch.tensor(b_terminal).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions_ = self.target_actor.forward(b_next_state)
        target_critic_value_ = self.target_critic.forward(b_next_state, target_actions_)
        critic_value = self.critic.forward(b_state, b_action)

        target = []
        for j in range(self.batch_size):
            target.append(b_reward[j] + self.gamma * target_critic_value_[j] * b_terminal[j])
        target = torch.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        b_mu = self.actor(b_state)
        self.actor.train()
        actor_loss = -self.critic.forward(b_state, b_mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_target_net_parameters()

    def update_target_net_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_dict = dict(actor_params)
        target_actor_dict = dict(target_actor_params)
        critic_dict = dict(critic_params)
        target_critic_dict = dict(target_critic_params)

        for name in actor_dict:
            actor_dict[name] = tau * actor_dict[name].clone() + (1 - tau) * target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_dict)

        for name in critic_dict:
            critic_dict[name] = tau * critic_dict[name].clone() + (1 - tau) * target_critic_dict[name].clone()
        self.target_critic.load_state_dict(critic_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()