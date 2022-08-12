import os

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow import keras
import random

from figure_sudoko_env_old import FigureSudokuEnv

env = FigureSudokuEnv()

# Global Variables
EPISODES = 500000

TRAIN_END = 0


# Hyper Parameters
def discount_rate():  # Gamma
    return 0.95


def learning_rate():  # Alpha
    return 0.001


def batch_size():  # Size of the batch used in the experience replay
    return 24


class DeepQNetwork:
    def __init__(self, states, actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=5000)
        self.alpha = alpha
        self.gamma = gamma
        # Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.loss = []
        self.model_filename = "dqn_model_weight.h5"

    def build_model(self):
        model = keras.Sequential()  # linear stack of layers https://keras.io/models/sequential/
        model.add(keras.layers.Dense(2048, input_dim=self.nS, activation='relu'))  # [Input] -> Layer 1
        model.add(keras.layers.Dense(1024, activation='relu'))  # [Input] -> Layer 1
        #   Dense: Densely connected layer https://keras.io/layers/core/
        #   24: Number of neurons
        #   input_dim: Number of input variables
        #   activation: Rectified Linear Unit (relu) ranges >= 0
        model.add(keras.layers.Dense(512, activation='relu'))  # Layer 2 -> 3
        model.add(keras.layers.Dense(self.nA, activation='linear'))  # Layer 3 -> [output]
        #   Size has to match the output (different actions)
        #   Linear activation on the last layer
        model.compile(loss='mean_squared_error',  # Loss function: Mean Squared Error
                      optimizer=keras.optimizers.Adam(lr=self.alpha))  # Optimizer: Adam (Feel free to check other options)
        return model

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA)  # Explore
        action_vals = self.model.predict(state)  # Exploit: Use the NN to predict the correct action from this state
        return np.argmax(action_vals[0])

    def test_action(self, state):  # Exploit
        action_vals = self.model.predict(state)
        return np.argmax(action_vals[0])

    def store(self, state, action, reward, nstate, done):
        # Store the experience in memory
        self.memory.append((state, action, reward, nstate, done))

    def experience_replay(self, batch_size):
        # Execute the experience replay
        minibatch = random.sample(self.memory, batch_size)  # Randomly sample from memory

        # Convert to numpy for speed by vectorization
        x = []
        y = []
        np_array = np.array(minibatch)
        st = np.zeros((0, self.nS))  # States
        nst = np.zeros((0, self.nS))  # Next States
        for i in range(len(np_array)):  # Creating the state and next state np arrays
            st = np.append(st, np_array[i, 0], axis=0)
            nst = np.append(nst, np_array[i, 3], axis=0)
        st_predict = self.model.predict(st)  # Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        index = 0
        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            # Predict from state
            nst_action_predict_model = nst_predict[index]
            if done == True:  # Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward
            else:  # Non terminal
                target = reward + self.gamma * np.amax(nst_action_predict_model)
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1
        # Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size, self.nS)
        y_reshape = np.array(y)
        epoch_count = 1  # Epochs is the number or iterations
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
        # Graph Losses
        for i in range(epoch_count):
            self.loss.append(hist.history['loss'][i])
        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        self.model.save(self.model_filename)

    def load_model(self):
        if os.path.isfile(self.model_filename):
            self.model.load_weights(self.model_filename)


# Create the agent
nS = env.num_inputs  # This 16 (4x4) fields
nA = env.num_actions  # Actions
dqn = DeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1.0, 0.01, 0.999999)

batch_size = batch_size()

# Training
rewards = []  # Store rewards for graphing
epsilons = []  # Store the Explore/Exploit
TEST_Episodes = 0
best_rewards = -999999
for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, nS]) # Resize to store in memory to pass to .predict
    tot_rewards = 0
    for time in range(50):  # 200 is when you "solve" the game. This can continue forever as far as I know
        action = dqn.action(state)
        nstate, reward, done = env.step(action)
        nstate = np.reshape(nstate, [1, nS])
        tot_rewards += reward
        dqn.store(state, action, reward, nstate, done) # Resize to store in memory to pass to .predict
        state = nstate
        # done: CartPole fell.
        # time == 209: CartPole stayed upright
        if done or time == 49:
            rewards.append(tot_rewards)
            epsilons.append(dqn.epsilon)
            print(f"episode: {e}/{EPISODES}, score: {tot_rewards}, e: {dqn.epsilon}")
            break

        # Experience Replay
        if len(dqn.memory) > batch_size:
            dqn.experience_replay(batch_size)

    if tot_rewards > best_rewards:
        dqn.save_model()
        best_rewards = tot_rewards

    # If our current NN passes we are done I am going to use the last 5 runs
    if len(rewards) > 5 and np.average(rewards[-5:]) > 15:
        # Set the rest of the EPISODES for testing
        TEST_Episodes = EPISODES - e
        TRAIN_END = e
        break

