import numpy as np
from matplotlib import pyplot as plt


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
