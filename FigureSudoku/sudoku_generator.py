import itertools
import random
import numpy as np
from figure_sudoko_env import Geometry, Color

geometries = np.array([Geometry.CIRCLE, Geometry.QUADRAT, Geometry.TRIANGLE, Geometry.HEXAGON])
colors = np.array([Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW])
rows = len(geometries)
cols = len(colors)

figures = np.array(list(itertools.product(geometries, colors)))
fields = np.array(list(itertools.product(np.arange(rows), np.arange(cols))))

actions = np.array(list(itertools.product(figures, fields)), dtype=object)
state = np.array([x for x in [[(Geometry.EMPTY.value, Color.EMPTY.value)] * rows] * cols])

possibilities = np.array([x for x in [[(geometries, colors)] * rows] * cols])

# random select first cell, geometry and color
row = random.randint(0, rows - 1)
col = random.randint(0, cols - 1)
geometry = random.choice(geometries)
color = random.choice(colors)

n = 0
max = 16
while True:  # n < max:

    state[row][col] = np.array([geometry.value, color.value])
    possibilities[row][col][0] = [Geometry.EMPTY, Geometry.EMPTY, Geometry.EMPTY, Geometry.EMPTY]
    possibilities[row][col][1] = [Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY]

    for i in range(cols):
        possibilities[row][i][0][possibilities[row][i][0] == geometry] = Geometry.EMPTY
        possibilities[row][i][1][possibilities[row][i][1] == color] = Color.EMPTY

    for i in range(rows):
        possibilities[i][col][0][possibilities[i][col][0] == geometry] = Geometry.EMPTY
        possibilities[i][col][1][possibilities[i][col][1] == color] = Color.EMPTY

    if np.all(possibilities[::, 0] == Geometry.EMPTY):
        break

    min_lg = len(geometries)
    min_lc = len(colors)
    for r in range(rows):
        for c in range(cols):
            lg = np.sum(possibilities[r][c][0] != Geometry.EMPTY)
            if 0 < lg <= min_lg:
                min_lg = lg

            lc = np.sum(possibilities[r][c][1] != Color.EMPTY)
            if 0 < lc <= min_lc:
                min_lc = lc

    cells = []
    for r in range(rows):
        for c in range(cols):
            if np.sum(possibilities[r][c][0] != Geometry.EMPTY) == min_lg and np.sum(possibilities[r][c][1] != Color.EMPTY) == min_lc:
                cells.append((r, c))

    if len(cells) < 1:
        break

    next_cell = random.choice(cells)
    (row, col) = next_cell

    print(possibilities[row][col][0][(possibilities[row][col][0] != Geometry.EMPTY)])
    print(possibilities[row][col][1][(possibilities[row][col][1] != Color.EMPTY)])
    geometry = possibilities[row][col][0][(possibilities[row][col][0] != Geometry.EMPTY)][0]
    color = possibilities[row][col][1][(possibilities[row][col][1] != Color.EMPTY)][0]

    n += 1

print(state)
