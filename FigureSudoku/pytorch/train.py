from collections import deque
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from figure_sudoko_env import FigureSudokuEnv
from pytorch.dqn_agent import Agent
from shapes import Geometry, Color

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU available: " + str(torch.cuda.is_available()))


TARGET_SCORE = 10.0


def train_sudoku(gui, stop):
    # create environment
    geometries = np.array([Geometry.CIRCLE, Geometry.QUADRAT, Geometry.TRIANGLE, Geometry.HEXAGON])
    colors = np.array([Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW])
    env = FigureSudokuEnv(geometries, colors, gui=gui)
    state_size = env.num_inputs
    action_size = env.num_actions

    agent = Agent(device=device, state_size=state_size, action_size=action_size, seed=0, double_dqn=False, dueling_network=True, prioritized_replay=True)

    weights_file = 'weights/checkpoint.pth'
    agent.load(weights_file)

    # hyperparameter
    max_episodes = 1000000
    max_timesteps = 250
    eps_start = 0.8
    eps_end = 0.01
    eps_decay = 0.999995
    start_level = 2

    # score parameter
    window_size = 100
    warmup_episodes = 2 * window_size
    scores_deque = deque(maxlen=window_size)
    avg_score = -99999
    best_avg_score = avg_score

    eps = eps_start  # initialize epsilon
    level = start_level

    writer = SummaryWriter()

    for episode in range(1, max_episodes + 1):
        if stop():
            break

        state = env.reset(level=level)  # reset the environment
        episode_score = 0
        for timestep in range(1, max_timesteps + 1):
            #possible_actions = env.get_possible_actions(state)
            action = agent.act(state, eps)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_score += reward
            if done:
                # print(f'Episode {episode:06d} - Step {timestep:06d}\tEpisode Score: {episode_score:.2f}\tdone!')
                break

        # average score over the last n epochs
        scores_deque.append(episode_score)
        avg_score = np.mean(scores_deque)

        writer.add_scalar("episode score", episode_score, episode)
        writer.add_scalar("avg score", avg_score, episode)
        writer.add_scalar("epsilon", eps, episode)

        if episode % 100 == 0:
            print(f'\rEpisode {episode:08d}\tAverage Score: {avg_score:.2f}\tEpsilon: {eps:.8f}')

        # print(f'Episode {episode:06d}\tAvg Score: {avg_score:.2f}\tBest Avg Score: {best_avg_score:.2f}')

        #eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        # save best weights
        if episode > warmup_episodes and avg_score > best_avg_score:
            best_avg_score = avg_score
            agent.save(weights_file)
            print(f'Episode {episode:08d}\tWeights saved!\tBest Avg Score: {best_avg_score:.2f}')

        # stop training if target score was reached
        if episode > warmup_episodes and avg_score >= TARGET_SCORE:
            agent.save(weights_file)
            print(f'\nEnvironment solved in {episode} episodes!\tAverage Score: {avg_score:.2f}')
            break
