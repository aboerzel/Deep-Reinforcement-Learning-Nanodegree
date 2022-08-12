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
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(env.num_actions, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

MODEL_FILE = 'model3.h5'


def save_model_weights():
    # Save trained weights
    model.save_weights(MODEL_FILE)


def load_model_weights():
    if pathlib.Path(MODEL_FILE).is_file():
        model.load_weights(MODEL_FILE)


#load_model_weights()

TARGET_SCORE = 28
gamma = 0.975
eps = 0.5
eps_min = 0.05
eps_decay_factor = 0.9999

scores_deque = deque(maxlen=100)
best_avg_score = -99999
episode = 0

while True:
    state = env.reset()  # using the harder state initialization function
    # decrement epsilon over time
    episode += 1
    episode_reward = 0
    # while game still in progress
    while True:
        # We are in state S
        # Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1, env.num_inputs), batch_size=1)
        if random.random() < eps:  # choose random action
            action = np.random.randint(0, env.num_actions)
        else:  # choose best action from Q(s,a) values
            action = (np.argmax(qval))
        # Take action, observe new state S'
        new_state, reward, done = env.step(action)

        # Get max_Q(S',a)
        # old_qval = model.predict(state.reshape(1, env.num_inputs), batch_size=1)
        newQ = model.predict(new_state.reshape(1, env.num_inputs), batch_size=1)
        maxQ = np.max(newQ)

        if reward == 1:  # non-terminal state
            update = reward + (gamma * maxQ)
        else:  # terminal state
            update = reward

        X = state.reshape(env.num_inputs, )
        y = np.zeros((1, env.num_actions))
        y[:] = qval[:]
        y[0][action] = update
        y = y.reshape(env.num_actions, )

        X_train = np.array([X])
        y_train = np.array([y])
        print(f'Episode {episode:06d}\tScore: {reward:.2f}\teps: {eps:.8f}')
        model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=1)
        state = new_state

        episode_reward += reward

        if done:
            print(f'Episode {episode:06d}\tEpisode Score: {episode_reward:.2f}\tdone!')
            break

        if reward != 1:  # if reached terminal state, update game
            print(f'Episode {episode:06d}\tEpisode Score: {episode_reward:.2f}\tfailed!')
            break

    # average score over the last n epochs
    scores_deque.append(episode_reward)
    avg_score = np.mean(scores_deque)

    print(f'Episode {episode:06d}\tAvg Score: {avg_score:.2f}\tBest Avg Score: {best_avg_score:.2f}')

    if eps > eps_min:
        eps *= eps_decay_factor

    # save best weights
    if avg_score > best_avg_score:
        best_avg_score = avg_score
        save_model_weights()
        print(f'\rEpisode {episode:06d}\tWeights saved!\tBest Avg Score: {best_avg_score:.2f}')

    print('\n')

    # stop training if target score was reached
    if avg_score >= TARGET_SCORE:
        save_model_weights()
        print(f'\nEnvironment solved in {episode:06d} episodes!\tAverage {len(scores_deque):d}\tScore: {avg_score:.2f}')
        break

