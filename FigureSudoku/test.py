import pathlib
import random
from collections import deque

import numpy as np
from keras import Input
from keras.models import Sequential
from keras.layers.core import Dense
from figure_sudoko_env import FigureSudokuEnv

env = FigureSudokuEnv()

model = Sequential()
model.add(Input(shape=env.num_inputs))
model.add(Dense(512, activation='relu'))
model.add(Dense(env.num_actions, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

MODEL_FILE = 'model2.h5'


def save_model_weights():
    # Save trained weights
    model.save_weights(MODEL_FILE)


def load_model_weights():
    if pathlib.Path(MODEL_FILE).is_file():
        model.load_weights(MODEL_FILE)


load_model_weights()

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

        if reward == -10:
            print(f'\nEpisode {episode:06d} fail!')
            break

    print(f'\rEpisode {episode:06d} - Score: {episode_reward}')

    # average score over the last n epochs
    scores_deque.append(episode_reward)
    avg_score = np.mean(scores_deque)

    if avg_score > best_avg_score:
        best_avg_score = avg_score
        print(f'\rEpisode {episode:06d} - Best Score: {best_avg_score:.2f}')
