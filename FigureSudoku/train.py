import pathlib
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.gradient_descent import SGD
from keras.regularizers import l2

from figure_sudoko_env import FigureSudokuEnv
from shapes import Geometry, Color
from torch.utils.tensorboard import SummaryWriter


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.learning_rate = 0.001
        self.tau = .08
        self.decay_rate = 0.001
        self.primary_network = self.build_model()
        self.target_network = self.build_model()

    def build_model(self):
        factor = 0.0001
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation="relu", activity_regularizer=l2(factor)))
        #model.add(Dense(128, activation="relu", activity_regularizer=l2(factor)))
        model.add(Dense(64, activation="relu", activity_regularizer=l2(factor)))
        model.add(Dense(self.action_size, activation='linear'))
        #model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate, decay=self.decay_rate))
        #model.compile(loss='mse', optimizer=SGD(learning_rate=self.learning_rate))
        model.compile(loss='mse', optimizer=Adam())
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, possible_actions, epsilon):
        if np.random.rand() <= epsilon:
            return random.choice(possible_actions)
        act_values = self.primary_network.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        minibatch = random.sample(self.memory, batch_size)
        x_batch, y_batch = [], []
        for state, action, reward, next_state, done in minibatch:
            y_target = self.primary_network.predict(state)
            y_target[0][action] = reward + (1-done) * self.gamma * np.amax(self.target_network.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        history = self.primary_network.fit(np.array(x_batch), np.array(y_batch), batch_size=batch_size, verbose=0)

        for t, e in zip(self.target_network.trainable_variables, self.primary_network.trainable_variables):
            t.assign(t * (1 - self.tau) + e * self.tau)

        # Keeping track of loss
        loss = history.history['loss'][0]
        return loss

    #def target_train(self):
    #    # update the target network with new weights
    #    self.target_network.set_weights(self.primary_network.get_weights())#
    #
    #    weights = self.primary_network.get_weights()
    #    target_weights = self.target_network.get_weights()
    #    for i in range(len(target_weights)):
    #        target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
    #    self.target_network.set_weights(target_weights)

    def load(self, name):
        if pathlib.Path(name).is_file():
            self.primary_network.load_weights(name)
            self.target_network.load_weights("target.h5")

    def save(self, name):
        self.primary_network.save_weights(name)
        self.target_network.save_weights("target.h5")


#if __name__ == "__main__":
def train_sudoku(gui, stop):
    geometries = np.array([Geometry.CIRCLE, Geometry.QUADRAT, Geometry.TRIANGLE, Geometry.HEXAGON])
    colors = np.array([Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW])

    env = FigureSudokuEnv(geometries, colors, gui=gui)
    state_size = env.num_inputs
    action_size = env.num_actions
    agent = DQNAgent(state_size, action_size)

    MODEL_NAME = 'model.h5'

    agent.load(MODEL_NAME)

    writer = SummaryWriter()

    BATCH_SIZE = 32
    EPSILON = 0.9
    #EPSILON = 0.01
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    TARGET_SCORE = 249
    UPDATE_EVERY = 10
    MAX_TIMESTEPS = 250
    START_LEVEL = 1
    LEVEL = START_LEVEL

    window_size = 100
    warmup_episodes = window_size * 2
    scores_deque = deque(maxlen=window_size)
    avg_score = -99999
    best_avg_score = avg_score
    episode = 0

    while True:
        if stop():
            break

        state = env.reset(level=LEVEL)
        state = np.reshape(state, [1, state_size])
        episode += 1
        episode_reward = 0
        avg_loss = 0
        timestep = 0

        print(f'Episode {episode:06d}\teps: {EPSILON:.8f}')

        #for timestep in range(1, MAX_TIMESTEPS+1):
        while True:
            timestep += 1
            possible_actions = env.get_possible_actions(state)
            action = agent.act(state, possible_actions, EPSILON)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            loss = agent.replay(BATCH_SIZE)  # internally iterates default (prediction) model

            state = next_state

            if loss is not None:
                avg_loss += loss

            episode_reward += reward

            if done:
                avg_loss /= timestep
                # print(f"\nEpisode: {episode:06d}, score: {timestep}, eps: {agent.epsilon:.8f}")
                writer.add_scalar("reward", episode_reward, episode)
                if loss is not None:
                    writer.add_scalar("loss", loss, episode)
                    writer.add_scalar("avg loss", avg_loss, episode)
                print(f'Episode {episode:06d} - Step {timestep:06d}\tEpisode Score: {episode_reward:.2f}\tdone!')
                break

            if timestep % UPDATE_EVERY == 0:
                print(f'Episode {episode:06d} - Step {timestep:06d}\tEpisode Score: {episode_reward:.2f}')



        # average score over the last n epochs
        scores_deque.append(episode_reward)
        avg_score = np.mean(scores_deque)

        writer.add_scalar("avg reward", avg_score, episode)
        writer.add_scalar("epsilon", EPSILON, episode)

        if loss is not None:
            print(f'Episode {episode:06d}\tAvg Score: {avg_score:.2f}\tBest Avg Score: {best_avg_score:.2f}\tLoss: {loss:.06f}\tAvg Loss: {avg_loss:.5f}')
        else:
            print(
                f'Episode {episode:06d}\tAvg Score: {avg_score:.2f}\tBest Avg Score: {best_avg_score:.2f}')

        # update epsilon
        if episode > warmup_episodes and avg_score > best_avg_score:
            if EPSILON > EPSILON_MIN:
                EPSILON *= EPSILON_DECAY

        # save best weights
        if episode > warmup_episodes and EPSILON < 0.1 and avg_score > best_avg_score and len(agent.memory) > BATCH_SIZE:
            best_avg_score = avg_score
            agent.save(MODEL_NAME)
            print(f'Episode {episode:06d}\tWeights saved!\tBest Avg Score: {best_avg_score:.2f}')

        print('\n')

        # stop training if target score was reached
        if episode > warmup_episodes and EPSILON < 0.1 and avg_score >= TARGET_SCORE:
            agent.save(MODEL_NAME)
            print(f'Environment solved in {episode:06d} episodes!\tAverage {len(scores_deque):d}\tScore: {avg_score:.2f}')
            break
