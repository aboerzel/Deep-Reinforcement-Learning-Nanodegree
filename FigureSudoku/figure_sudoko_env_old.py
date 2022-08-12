import itertools
import random

import numpy as np


class FigureSudokuEnv:
    colors = ['red', 'green', 'blue', 'yellow']
    geometries = ['square', 'circle', 'triangle', 'hexagon']

    def __init__(self, rows=4, cols=4):
        self.figures = np.array(list(itertools.product(FigureSudokuEnv.geometries, FigureSudokuEnv.colors)))
        fields = np.array(list(itertools.product(np.arange(rows), np.arange(cols))))
        self.actions = np.array(list(itertools.product(range(len(self.figures)), fields)), dtype=object)
        self.state = [x[:] for x in [[-1] * rows] * cols]

        self.num_inputs = len(fields)
        self.num_actions = len(self.actions)

    def reset(self):
        self.state = [x[:] for x in [[-1] * len(self.state[0])] * len(self.state[1])]
        row = random.randint(0, len(self.state[0])-1)
        col = random.randint(0, len(self.state[1])-1)
        self.state[row][col] = random.randint(0, len(self.figures)-1)
        return np.array(self.state).flatten()

    def step(self, action):
        target_action = self.actions[action]

        figure_idx = target_action[0]
        figure = self.figures[figure_idx]
        geometry = figure[0]
        color = figure[1]
        (row, col) = target_action[1]

        if not FigureSudokuEnv.is_figure_available(self.state, figure_idx):
            return np.array(self.state).flatten(), -15, False

        if not FigureSudokuEnv.is_field_empty(self.state, row, col):
            return np.array(self.state).flatten(), -15, False

        if not FigureSudokuEnv.can_move(self.state, self.figures, row, col, geometry, color):
            return np.array(self.state).flatten(), -15, False

        self.state[row][col] = figure_idx
        done = FigureSudokuEnv.is_done(self.state)
        reward = 20 if done else 1
        return np.array(self.state).flatten(), reward, done

    #@staticmethod
    #def get_figure_index(figures, geometry, color):
    #    return np.where(np.logical_and(figures[:, 0] == geometry, figures[:, 1] == color))[0][0]

    @staticmethod
    def is_field_empty(state, row, col):
        return state[row][col] == -1

    @staticmethod
    def is_figure_available(state, figure_idx):
        return len(np.where(np.array(state).flatten() == figure_idx)[0]) == 0

    @staticmethod
    def is_done(state):
        return len(np.where(np.array(state).flatten() == -1)[0]) == 0

    @staticmethod
    def can_move(state, figures, row, col, geometry, color):
        for figure_idx in state[row]:
            if figure_idx == -1:
                continue
            figure = figures[figure_idx]
            if figure[0] == geometry:
                return False
            if figure[1] == color:
                return False

        for figure_idx in np.array(state)[:, col]:
            if figure_idx == -1:
                continue
            figure = figures[figure_idx]
            if figure[0] == geometry:
                return False
            if figure[1] == color:
                return False

        return True
