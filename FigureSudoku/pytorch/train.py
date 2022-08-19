from collections import deque
from dqn_agent import Agent
import os
import numpy as np
import matplotlib.pyplot as plt

from figure_sudoko_env import FigureSudokuEnv

# output folder for trained weights
os.makedirs("weights", exist_ok=True)

env = FigureSudokuEnv()
# number of actions
action_size = env.num_inputs
print('Number of actions:', action_size)

# examine the state space
state_size = env.num_actions
print('States have length:', state_size)

TARGET_SCORE = 10.0


def dqn(agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    mean_scores = []  # list the mean of the window scores
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()  # reset the environment
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            state = env.step(action)  # send the action to the environment
            next_state, reward, done = state  # get the next state
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores.append(score)  # save most recent score
        scores_window.append(score)  # save most recent score
        mean_score = np.mean(scores_window)  # mean score
        mean_scores.append(mean_score)  # save mean score

        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))

        if mean_score >= TARGET_SCORE:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, mean_score))
            break

    return scores, mean_scores


def train(agent, weights_file):
    scores, means = dqn(agent, n_episodes=100000, max_t=50, eps_start=0.10, eps_end=0.01, eps_decay=0.98)
    agent.save(weights_file)

    # plot the scores
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(means)), means, linestyle='--')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(('Score', 'Mean'), fontsize='xx-large')
    plt.show()


def evaluate(env, agent, episodes, weights_file):
    # load trained weights from file
    agent.load(weights_file)

    scores = []  # list containing scores from each episode

    for i_episode in range(1, episodes + 1):
        state = env.reset()  # reset the environment
        score = 0
        while True:
            action = agent.act(state)  # get the next action
            next_state, reward, done = env.step(action)  # send the action to the environment
            score += reward
            state = next_state
            print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, score), end="")
            if done:
                break

        scores.append(score)  # save most recent score

        print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, scores[-1]))

    mean_score = np.mean(scores)
    print('\nAverage Score over {} episodes: {:.2f}!'.format(episodes, mean_score))

    # plot the scores
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(scores)), [np.mean(scores)] * len(scores), linestyle='--')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(('Score', 'Mean'), fontsize='xx-large')
    plt.show()


agent = Agent(state_size=state_size, action_size=action_size, seed=0, double_dqn=False, dueling_network=False, prioritized_replay=False)

weights_file = 'weights/checkpoint.pth'
train(agent, weights_file)

