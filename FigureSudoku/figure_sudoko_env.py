import itertools
import random

import numpy as np
from enum import Enum


class Reward(Enum):
    FORBIDDEN = -1
    CONTINUE = 1
    DONE = 100


class Geometry(Enum):
    EMPTY = -1
    CIRCLE = 0
    QUADRAT = 1
    TRIANGLE = 2
    HEXAGON = 3


class Color(Enum):
    EMPTY = -1
    RED = 0
    GREEN = 1
    BLUE = 2
    YELLOW = 3


class FigureSudokuEnv:

    def __init__(self, rows=len(Geometry)-1, cols=len(Color)-1):
        self.rows = rows
        self.cols = cols
        self.geometries = np.array([Geometry.CIRCLE, Geometry.QUADRAT, Geometry.TRIANGLE, Geometry.HEXAGON])
        self.colors = np.array([Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW])
        figures = np.array(list(itertools.product(self.geometries, self.colors)))
        fields = np.array(list(itertools.product(np.arange(rows), np.arange(cols))))
        self.actions = np.array(list(itertools.product(figures, fields)), dtype=object)
        self.state = np.array([x for x in [[(Geometry.EMPTY.value, Color.EMPTY.value)] * rows] * cols])

        self.num_inputs = len(self.state.flatten())
        self.num_actions = len(self.actions)

    def reset(self):
        self.state = np.array([x for x in [[(Geometry.EMPTY.value, Color.EMPTY.value)] * self.rows] * self.cols])
        row = random.randint(0, self.rows-1)
        col = random.randint(0, self.cols-1)
        self.state[row][col] = np.array([random.choice(self.geometries).value, random.choice(self.colors).value])
        return self.state.flatten()

    def step(self, action):
        target_action = self.actions[action]

        (geometry, color) = target_action[0]
        (row, col) = target_action[1]

        if not FigureSudokuEnv.is_figure_available(self.state, geometry, color):
            return self.state.flatten(), Reward.FORBIDDEN.value, False

        if not FigureSudokuEnv.is_field_empty(self.state, row, col):
            return self.state.flatten(), Reward.FORBIDDEN.value, False

        if not FigureSudokuEnv.can_move(self.state, row, col, geometry, color):
            return self.state.flatten(), Reward.FORBIDDEN.value, False

        self.state[row][col] = [geometry.value, color.value]
        done = FigureSudokuEnv.is_done(self.state)
        reward = Reward.DONE.value if done else Reward.CONTINUE.value
        return self.state.flatten(), reward, done

    @staticmethod
    def is_field_empty(state, row, col):
        return state[row][col][0] == Geometry.EMPTY.value or state[row][col][1] == Color.EMPTY.value

    @staticmethod
    def is_figure_available(state, geometry, color):
        x = state.reshape(16, 2)
        return len(np.where(np.logical_and(x[:, 0] == geometry.value, x[:, 1] == color.value))[0]) == 0

    @staticmethod
    def is_done(state):
        return len(np.where(np.logical_or(state[:, 0] == Geometry.EMPTY.value, state[:, 1] == Color.EMPTY.value))[0]) == 0

    @staticmethod
    def can_move(state, row, col, geometry, color):
        for field in state[row]:
            if field[0] == geometry:
                return False
            if field[1] == color:
                return False

        for field in np.array(state)[:, col]:
            if field[0] == geometry:
                return False
            if field[1] == color:
                return False

        return True
