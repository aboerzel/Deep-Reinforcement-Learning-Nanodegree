import pathlib
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizer_v2.adam import Adam

from figure_sudoko_env import FigureSudokuEnv


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # discount rate
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, activation='relu'))
        #model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        if pathlib.Path(name).is_file():
            self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = FigureSudokuEnv()
    state_size = env.num_inputs
    action_size = env.num_actions
    agent = DQNAgent(state_size, action_size)

    MODEL_NAME = 'model4.h5'

    agent.load(MODEL_NAME)

    BATCH_SIZE = 32
    EPSILON = 0.65 # 0.9
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.9999
    TARGET_SCORE = 100
    UPDATE_EVERY = 10
    MAX_TIMESTEPS = 100

    scores_deque = deque(maxlen=100)
    best_avg_score = -99999
    episode = 0

    while True:
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        episode += 1
        episode_reward = 0

        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

        print(f'Episode {episode:06d}\teps: {EPSILON:.8f}')

        for timestep in range(1, MAX_TIMESTEPS+1):
            # env.render()
            action = agent.act(state, EPSILON)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state

            episode_reward += reward

            if done:
                # print(f"\nEpisode: {episode:06d}, score: {timestep}, eps: {agent.epsilon:.8f}")
                print(f'Episode {episode:06d} - Step {timestep:06d}\tEpisode Score: {episode_reward:.2f}\tdone!')
                break

            if len(agent.memory) > BATCH_SIZE and (timestep % UPDATE_EVERY == 0) and not done:
                print(f'Episode {episode:06d} - Step {timestep:06d}\tEpisode Score: {episode_reward:.2f}')
                agent.replay(BATCH_SIZE)

            #if reward != 1:  # if reached terminal state, update game
            #    print(f'\nEpisode {episode:06d} - Step {timestep:06d}\tEpisode Score: {episode_reward:.2f}\tfailed!')
            #    break

        # average score over the last n epochs
        scores_deque.append(episode_reward)
        avg_score = np.mean(scores_deque)

        print(f'Episode {episode:06d}\tAvg Score: {avg_score:.2f}\tBest Avg Score: {best_avg_score:.2f}')

        # save best weights
        if avg_score > best_avg_score and len(agent.memory) > BATCH_SIZE:
            best_avg_score = avg_score
            agent.save(MODEL_NAME)
            print(f'Episode {episode:06d}\tWeights saved!\tBest Avg Score: {best_avg_score:.2f}')

        print('\n')

        # stop training if target score was reached
        if avg_score >= TARGET_SCORE:
            agent.save(MODEL_NAME)
            print(f'Environment solved in {episode:06d} episodes!\tAverage {len(scores_deque):d}\tScore: {avg_score:.2f}')
            break
