from DDPG_LSTM1 import Agent
import gym
import numpy as np
from tqdm import tqdm
import os

if not os.path.exists('tmp/ddpg1'):
    os.makedirs('tmp/ddpg1')


env = gym.make('Pendulum-v1')
input_dims = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
EPOCH = 1000
resume = False

agent = Agent(lr_l=0.0005, lr_c=0.001, lr_a=0.0005, input_dims=input_dims, lstm_dims=100, tau=0.001, env=env,
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=n_actions)
if resume:
    agent.load_models()

np.random.seed(0)
score_history = []

bar = tqdm(range(EPOCH), desc='EPOCH')
for e in bar:
    bar.set_description(f'EPOCH {e+1:0>4d}')
    done = False
    epoch_score = 0
    state = env.reset()
    h0 = np.zeros(100)
    c0 = np.zeros(100)
    while not done:
        env.render()
        lstm_state, (h, c) = agent.state_in_lstm(state, (h0, c0))
        act = agent.choose_action(lstm_state)
        next_state, reward, done, _ = env.step(act)
        agent.remember(state, act, reward, next_state, int(done), (h0, c0), (h, c))
        agent.learn()
        epoch_score += reward
        state = next_state
        h0, c0 = h, c
        
    score_history.append(epoch_score)
    bar.set_postfix(epoch_score=f'{epoch_score:.3f}', game_100_mean=f'{np.mean(score_history[-100:]):.3f}')

    if e % 50 == 0:
        agent.save_models()

        eval_bar = tqdm(range(5), position=0)
        eval_bar.set_description(f'EVAL_EPOCH {e:0>4d}')
        eval_score = []
        for i in eval_bar:
            done = False
            state = env.reset()
            score = 0
            h0 = np.zeros(100)
            c0 = np.zeros(100)
            while not done:
                env.render()
                lstm_state, (h, c) = agent.state_in_lstm(state, (h0, c0))
                act = agent.choose_action(lstm_state, evaluate=True)
                next_state, reward, done, _ = env.step(act)
                score += reward
                state = next_state
                h0, c0 = h, c
            eval_score.append(score)
            eval_bar.set_postfix(mean_total_reward=np.mean(eval_score))
