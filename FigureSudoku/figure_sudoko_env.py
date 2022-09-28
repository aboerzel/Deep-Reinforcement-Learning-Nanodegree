import itertools
import numpy as np
from enum import Enum
from shapes import Geometry, Color
from sudoku_generator import SudokuGenerator


class Reward(Enum):
    FORBIDDEN = -10
    CONTINUE = -1
    DONE = 10


class FigureSudokuEnv:

    def __init__(self, geometries, colors, gui=None):
        self.geometries = geometries
        self.colors = colors
        self.gui = gui
        self.rows = len(self.geometries)
        self.cols = len(self.colors)
        self.figures = np.array(list(itertools.product(self.geometries, self.colors)))
        fields = np.array(list(itertools.product(np.arange(self.rows), np.arange(self.cols))))
        self.actions = np.array(list(itertools.product(self.figures, fields)), dtype=object)
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

    def get_possible_actions(self, state):
        state = state.reshape(16, 2)

        # get used figures
        used_figures = state[np.logical_and(state[:, 0] != Geometry.EMPTY.value, state[:, 1] != Color.EMPTY.value)]
        used_figures = [[Geometry(f[0]), Color(f[1])] for f in used_figures]

        # get used cells
        test = np.array([a for a in np.where(np.logical_and(state[:, 0] == Geometry.EMPTY.value, state[:, 1] == Color.EMPTY.value))]).squeeze(axis=0)

        used_cells = []
        for x in test:
            row = int(x / self.rows)
            col = x % self.cols
            used_cells.append([row, col])

        possible_actions = self.actions.copy()
        # filter out used figures
        possible_actions = [a for a in possible_actions if
                            len([f for f in used_figures if a[0][0] == f[0] and a[0][1] == f[1]]) == 0]
        # filter out used cells
        possible_actions = [a for a in possible_actions if
                            len([c for c in used_cells if a[1][0] == c[0] and a[1][1] == c[1]]) != 0]

        # get indices
        possible_actions_ind = np.where([True if len([a for a in possible_actions if
                                                      a[0][0] == b[0][0] and a[0][1] == b[0][1] and a[1][0] == b[1][0]
                                                        and a[1][1] == b[1][1]]) > 0 else False for b in self.actions])[0]

        return possible_actions_ind

    def step(self, action):
        target_action = self.actions[action]

        (geometry, color) = target_action[0]
        (row, col) = target_action[1]

        temp_state = [geometry.value, color.value]

        if not FigureSudokuEnv.is_figure_available(self.state, geometry, color):
            return self.state.flatten(), Reward.FORBIDDEN.value, False

        if not FigureSudokuEnv.is_field_empty(self.state, row, col):
            return self.state.flatten(), Reward.FORBIDDEN.value, False

        if not FigureSudokuEnv.can_move(self.state, row, col, geometry, color):
            return self.state.flatten(), Reward.FORBIDDEN.value, False

        self.state[row][col] = temp_state
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
