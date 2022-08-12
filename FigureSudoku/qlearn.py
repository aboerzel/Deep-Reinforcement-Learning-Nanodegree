import os
from collections import deque

import numpy as np
from keras import Input
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from figure_sudoko_env import FigureSudokuEnv

env = FigureSudokuEnv()

discount_factor = 0.95
eps = 0.8
eps_decay_factor = 0.999
num_episodes = 100000

model = Sequential()
model.add(Input(shape=env.num_inputs))
model.add(Dense(512, activation='relu'))
model.add(Dense(env.num_actions, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])


def save_model_weights():
    # Save trained weights
    model.save_weights('q-model.h5')


scores_deque = deque(maxlen=100)
episodic_scores = []
avg_scores = []
best_avg_score = -99999

for episode in range(1, num_episodes):
    state = env.reset()
    state = tf.expand_dims(state, 0)
    eps *= eps_decay_factor
    done = False
    episode_reward = 0
    for step in range(1, 500):
        if np.random.random() < eps:
            action = np.random.randint(0, env.num_actions)
        else:
            action = np.argmax(model.predict(state))
        new_state, reward, done = env.step(action)
        episode_reward += reward
        print(f'\rEpisode {episode:05d}\t Step {step:08d}\t Score: {episode_reward} \t eps: {eps:.5f}', end='')
        new_state = tf.expand_dims(new_state, 0)
        target = reward + discount_factor * np.max(model.predict(new_state))
        target_vector = model.predict(state)[0]
        target_vector[action] = target
        model.fit(state, target_vector.reshape(-1, env.num_actions), epochs=1, verbose=0)
        state = new_state

        if done:
            print(f'\nEpisode {episode:05d} done!')
            break

    scores_deque.append(episode_reward)
    avg_score = np.mean(scores_deque)
    avg_scores.append(avg_score)

    if episode % 10 == 0:
        print('\rEpisode {0:05d}\tAverage {2:d} Score: {1:.2f}'.format(episode, avg_score, 10))

    # save best weights
    if avg_score > best_avg_score:
        best_avg_score = avg_score
        save_model_weights()
        print('\rEpisode {0:05d}\tWeights saved! - Best Score: {1:.2f}'.format(episode, best_avg_score))


print("End")