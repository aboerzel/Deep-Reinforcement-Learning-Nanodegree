import os
import pathlib
import tensorflow as tf
import numpy as np

from keras import layers
from tensorflow import keras
from figure_sudoko_env import FigureSudokuEnv
from collections import deque


OUTPUT_DIR = os.path.join('output', 'model')
pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

MODEL_FILE = os.path.normpath(os.path.join(OUTPUT_DIR, 'model.h5'))

TARGET_SCORE = 30.0     # Goal score at which the problem is solved
AVG_SCORE_WINDOW = 100  # Sliding window size for average score calculation
PRINT_EVERY = 10        # Print score all N episodes

env = FigureSudokuEnv()

gamma = 0.99  # Discount factor for past rewards
eps = np.finfo(np.float32).eps.item()

inputs = layers.Input(shape=env.num_inputs)
hidden1 = layers.Dense(1024, activation="relu")(inputs)
hidden2 = layers.Dense(512, activation="relu")(hidden1)
action = layers.Dense(env.num_actions, activation="softmax")(hidden2)
critic = layers.Dense(1)(hidden2)

model = keras.Model(inputs=inputs, outputs=[action, critic])


def load_model_weights():
    if pathlib.Path(MODEL_FILE).is_file():
        model.load_weights(MODEL_FILE)


def save_model_weights():
    # Save trained weights
    model.save_weights(MODEL_FILE)


# Load weights from previous training, if any
#load_model_weights()

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()

action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode = 1
max_steps_per_episode = 10000
scores_deque = deque(maxlen=AVG_SCORE_WINDOW)
episodic_scores = []
#avg_scores = []
best_avg_score = -99999

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(env.num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            print(f'\rEpisode {episode:06d} - Step {timestep:06d}\tEpisode Score: {episode_reward:.2f}', end='')
            if done:
                break

        print(f'\nEpisode {episode:06d} - Step {timestep:06d}\tEpisode Score: {episode_reward:.2f}\t{"done" if done else "failed"}!')

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        print(f'Episode {episode:06d}\tRunning Reward: {running_reward:.2f}')

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up receiving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of the future rewards.
            critic_losses.append(huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode += 1
    episodic_scores.append(episode_reward)
    # calculate mean score over last N episodes
    scores_deque.append(episode_reward)
    avg_score = np.mean(scores_deque)
    #avg_scores.append(avg_score)

    print(f'Episode {episode:06d}\tEpisode Score: {episode_reward:.2f}\t{"done" if done else "failed"}!')
    print(f'Episode {episode:06d}\tAvg Score: {avg_score:.2f}\tBest Avg Score: {best_avg_score:.2f}')

    # save best weights
    if avg_score > best_avg_score:
        best_avg_score = avg_score
        save_model_weights()
        print(f'\rEpisode {episode:06d}\tWeights saved!\tBest Avg Score: {best_avg_score:.2f}')

    # stop training if target score was reached
    if avg_score >= TARGET_SCORE:
        print(f'\nEnvironment solved in {episode:06d} episodes!\tAverage {len(scores_deque):d}\tScore: {avg_score:.2f}')
        break
