import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from tqdm import tqdm
import os

BATCH_SIZE = 100
LR = 0.1
EPSILON = 0.5
GAMMA = 0.99
TARGET_UPDATE_ITER = 100
MEMORY_CAPACITY = 30000
MEMORY_WARMUP_CAPACITY = 1000
EPOCH = 3000
resume = False

weight = None
if not os.path.exists('./weights'):
    os.makedirs('./weights')
if os.path.exists('./weights/weight0000.pkl'):
    weight = torch.load('./weights/weight0000.pkl')


env = gym.make('CartPole-v0')
env = env.unwrapped
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
env_a_shape = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_states, 20)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(20, num_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        Q_value = self.out(x)
        return Q_value


class DQN(object):
    def __init__(self):
        self.eval_net = Net()
        self.target_net = Net()

        self.learn_step = 0
        self.memory_count = 0
        self.memory = np.zeros((MEMORY_CAPACITY, num_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, ep):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # ???

        if np.random.uniform() < ep:
            action = np.random.randint(0, num_actions)
            action = action if env_a_shape == 0 else action.reshape(env_a_shape)
        else:
            Q_value = self.eval_net.forward(x)
            action = torch.max(Q_value, 1)[1].data.numpy()
            action = action[0] if env_a_shape == 0 else action.reshape(env_a_shape)

        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_count % MEMORY_CAPACITY
        self.memory[index] = transition
        self.memory_count += 1

    def learn(self):
        if self.learn_step % TARGET_UPDATE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        sample_index = np.random.choice(self.memory_count if self.memory_count < MEMORY_CAPACITY else MEMORY_CAPACITY,
                                        BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :num_states])
        b_a = torch.LongTensor(b_memory[:, num_states:num_states + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, num_states + 1:num_states + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -num_states:])

        Q_eval = self.eval_net(b_s).gather(1, b_a)
        Q_next = self.target_net(b_s_).detach()
        Q_target = b_r + GAMMA * Q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(Q_eval, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        return loss


dqn = DQN()
if resume and weight:
    dqn.eval_net.load_state_dict(weight)

bar = tqdm(range(EPOCH), desc='EPOCH')
for e in bar:
    bar.set_description(f'EPOCH {e:0>4d}')
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_action(s, EPSILON)

        s_, r, done, info = env.step(a)
        if done:
            r = -100

        # x, x_dot, theta, theta_dot = s_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # r = r1 + r2
        dqn.store_transition(s, a, r, s_)
        ep_r += r

        if dqn.memory_count > MEMORY_WARMUP_CAPACITY:
            loss = dqn.learn()

        if done or ep_r > 1000:
            # print(f'EPOCH: {e}, EP_R: {ep_r:.3f}, LR: {LR:.3f}, EPSILON: {EPSILON:.3f}')
            break

        s = s_

    if e % 50 == 0:
        eval_reward = []
        eval_bar = tqdm(range(5), position=0)
        eval_bar.set_description(f'EVAL_EPOCH {e:0>4d}')
        for i in eval_bar:
            s = env.reset()
            ep_r = 0
            while True:
                env.render()
                a = dqn.choose_action(s, 0)
                s_, r, done, info = env.step(a)
                ep_r += r
                s = s_
                if done or ep_r > 1000:
                    break
            eval_reward.append(ep_r)
            eval_bar.set_postfix(mean_total_reward=np.mean(eval_reward))
        if np.mean(eval_reward) > 500:
            path = f'./weights/good_weight{e:0>4d}r{np.mean(eval_reward)}.pkl'
            torch.save(dqn.eval_net.state_dict(), path)
            print(f'\nweight save in {os.getcwd()}{path[1:]}.')
        elif e % 500 == 0:
            path = f'./weights/weight{e:0>4d}r{np.mean(eval_reward)}.pkl'
            torch.save(dqn.eval_net.state_dict(), path)
            print(f'\nweight save in {os.getcwd()}{path[1:]}.')

    if dqn.memory_count > MEMORY_WARMUP_CAPACITY:
        if EPSILON > 0.01:
            EPSILON *= 0.995
        if LR > 0.001:
            LR *= 0.996
        bar.set_postfix(LR=LR, epsilon=EPSILON, e_r=ep_r, loss=loss)

