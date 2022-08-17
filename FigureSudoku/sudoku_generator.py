import itertools
import random
import numpy as np
from figure_sudoko_env import Geometry, Color

geometries = np.array([Geometry.CIRCLE, Geometry.QUADRAT, Geometry.TRIANGLE, Geometry.HEXAGON])
colors = np.array([Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW])
rows = len(geometries)
cols = len(colors)
state_size = rows * cols

shapes = list(itertools.product(geometries, colors))

solved = False

while not solved:

    state = np.array([x for x in [[(Geometry.EMPTY.value, Color.EMPTY.value)] * rows] * cols])
    possibilities = [[shapes for i in range(cols)] for j in range(rows)]

    # random select first cell, geometry and color
    row = random.randint(0, rows - 1)
    col = random.randint(0, cols - 1)
    geometry = random.choice(geometries)
    color = random.choice(colors)

    while True:
        state[row][col] = np.array([geometry.value, color.value])
        possibilities[row][col] = []

        for r in range(rows):
            for c in range(cols):
                possibilities[r][c] = [item for item in possibilities[r][c] if not (item[0] == geometry and item[1] == color)]

        for i in range(cols):
            possibilities[row][i] = [item for item in possibilities[row][i] if item[0] != geometry and item[1] != color]

        for i in range(rows):
            possibilities[i][col] = [item for item in possibilities[i][col] if item[0] != geometry and item[1] != color]

        min_length = 999
        for r in range(rows):
            for c in range(cols):
                length = len(possibilities[r][c])
                if 0 < length <= min_length:
                    min_length = length

        cells = []
        for r in range(rows):
            for c in range(cols):
                if len(possibilities[r][c]) == min_length:
                    cells.append((r, c))

        if len(cells) < 1:
            break

        (row, col) = random.choice(cells)
        possible_shapes = possibilities[row][col]
        (geometry, color) = random.choice(possible_shapes)
        #print(geometry, color)

    solved = np.all(state.reshape(state_size, 2)[:, 0] != Geometry.EMPTY.value) and np.all(state.reshape(state_size, 2)[:, 1] != Color.EMPTY.value)
    #print(f'solved: {solved}')

print("final state:")
print(state)
print()
print("init state:")
init_state = state.reshape(state_size, 2)
idx = np.random.choice(range(state_size), 3, replace=False)
init_state = init_state[idx]
print(init_state)



