from collections import deque

import numpy as np
from figure_sudoko_env import FigureSudokuEnv
from train import DQNAgent

env = FigureSudokuEnv()

model = DQNAgent(state_size=env.num_inputs, action_size=env.num_actions).build_model()
model.load_weights('model4.h5')

episodes = 10
scores_deque = deque(maxlen=100)
best_avg_score = -99999

for episode in range(1, episodes):
    state = env.reset()
    episode_reward = 0

    while True:
        qval = model.predict(state.reshape(1, env.num_inputs), batch_size=1)
        action = (np.argmax(qval))
        state, reward, done = env.step(action)

        episode_reward += reward

        if done:
            print(f'\nEpisode {episode:06d} done!')
            break

        if reward == -1:
            print(f'\nEpisode {episode:06d} fail!')
            break

    print(f'\rEpisode {episode:06d} - Score: {episode_reward}')

    # average score over the last n epochs
    scores_deque.append(episode_reward)
    avg_score = np.mean(scores_deque)

    if avg_score > best_avg_score:
        best_avg_score = avg_score
        print(f'\rEpisode {episode:06d} - Best Score: {best_avg_score:.2f}')
