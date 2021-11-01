import gym
import numpy as np
from SAC_Agent import Agent
from tqdm import tqdm
import os


def check_dir_exist():
    if not os.path.exists('tmp/sac'):
        os.makedirs('tmp/sac')


if __name__ == '__main__':
    game_name = 'Lunar'
    check_dir_exist()
    env = gym.make('LunarLanderContinuous-v2')
    input_dims = env.observation_space.shape
    n_actions = env.action_space.shape[0]
    max_action = env.action_space.high
    agent = Agent(input_dims, n_actions, max_action, env)

    max_epoch = 1000

    best_score = env.reward_range[0]
    score_history = []
    load = False

    if load:
        agent.load_models()

    bar = tqdm(range(max_epoch), desc='EPOCH', position=0)
    for e in bar:
        bar.set_description(f'EPOCH {e + 1:0>4d}')
        done = False
        epoch_score = 0
        state = env.reset()
        while not done:
            env.render()
            action = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            epoch_score += reward
            agent.remember(state, action, reward, state_, done)
            if not load:
                agent.learn()
            state = state_
        score_history.append(epoch_score)
        avg_score = np.mean(score_history[-100:])
        bar.set_postfix(e_score=f'{epoch_score:.3f}', mean=f'{avg_score:.3f}', best=f'{best_score:.3f}')
        if avg_score > best_score:
            best_score = avg_score
            if not load:
                agent.save_models(n=game_name+'_best')

        if e % 50 == 0:
            agent.save_models(n=game_name+'_last')
