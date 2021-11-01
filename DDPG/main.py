from DDPG import Agent
import gym
import numpy as np
from tqdm import tqdm
import os

if not os.path.exists('tmp/ddpg'):
    os.makedirs('tmp/ddpg')


env = gym.make('LunarLanderContinuous-v2')
input_dims = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
EPOCH = 1000
resume = True

agent = Agent(lr_c=0.0025, lr_a=0.001, input_dims=[input_dims], tau=0.001, env=env,
              batch_size=16, layer1_size=400, layer2_size=300, n_actions=n_actions)
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
    while not done:
        env.render()
        act = agent.choose_action(state)
        next_state, reward, done, _ = env.step(act)
        agent.remember(state, act, reward, next_state, int(done))
        agent.learn()
        epoch_score += reward
        state = next_state
        
    score_history.append(epoch_score)
    bar.set_postfix(epoch_score=f'{epoch_score:.3f}', game_100_mean=f'{np.mean(score_history[-100:]):.3f}')

    if e % 50 == 0:
        # agent.save_models()

        eval_bar = tqdm(range(5), position=0)
        eval_bar.set_description(f'EVAL_EPOCH {e:0>4d}')
        eval_score = []
        for i in eval_bar:
            done = False
            state = env.reset()
            score = 0
            while not done:
                env.render()
                act = agent.choose_action(state, evaluate=True)
                next_state, reward, done, _ = env.step(act)
                score += reward
                state = next_state
            eval_score.append(score)
            eval_bar.set_postfix(mean_total_reward=np.mean(eval_score))
