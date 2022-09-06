import itertools
import random
import numpy as np
from enum import Enum
from shapes import Geometry, Color
from sudoku_generator import SudokuGenerator


class Reward(Enum):
    FORBIDDEN = -1
    CONTINUE = 0
    DONE = 250


class FigureSudokuEnv:

    def __init__(self, geometries, colors, gui=None):
        self.geometries = geometries
        self.colors = colors
        self.gui = gui
        self.rows = len(self.geometries)
        self.cols = len(self.colors)
        figures = np.array(list(itertools.product(self.geometries, self.colors)))
        fields = np.array(list(itertools.product(np.arange(self.rows), np.arange(self.cols))))
        self.actions = np.array(list(itertools.product(figures, fields)), dtype=object)
        self.state = np.array([x for x in [[(Geometry.EMPTY.value, Color.EMPTY.value)] * self.rows] * self.cols])

        self.num_inputs = len(self.state.flatten())
        self.num_actions = len(self.actions)

        self.generator = SudokuGenerator(geometries, colors)

    def reset(self, level=1):
        initial_items = (self.rows * self.cols) - level
        self.state = self.generator.generate(initial_items=initial_items)[1]

        # randomly occupy a cell with a figure
        #self.state = np.array([x for x in [[(Geometry.EMPTY.value, Color.EMPTY.value)] * self.rows] * self.cols])
        #figure, field = random.choice(self.actions)
        #(row, col) = field
        #(geometry, color) = figure
        #self.state[row][col][0] = geometry.value
        #self.state[row][col][1] = color.value

        # update gui
        if self.gui is not None:
            self.gui.display_state(self.state)
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

        if self.gui is not None:
            self.gui.display_state(self.state)

        return self.state.flatten(), reward, done

    @staticmethod
    def is_field_empty(state, row, col):
        return state[row][col][0] == Geometry.EMPTY.value or state[row][col][1] == Color.EMPTY.value

    @staticmethod
    def is_figure_available(state, geometry, color):
        state = state.reshape(state.shape[0] * state.shape[1], 2)
        return len(np.where(np.logical_and(state[:, 0] == geometry.value, state[:, 1] == color.value))[0]) == 0

    @staticmethod
    def is_done(state):
        state = state.reshape(state.shape[0] * state.shape[1], 2)
        return len(np.where(np.logical_or(state[:, 0] == Geometry.EMPTY.value, state[:, 1] == Color.EMPTY.value))[0]) == 0

    @staticmethod
    def can_move(state, row, col, geometry, color):
        for field in state[row]:
            if field[0] == geometry.value:
                return False
            if field[1] == color.value:
                return False

        for field in np.array(state)[:, col]:
            if field[0] == geometry.value:
                return False
            if field[1] == color.value:
                return False

        return True
