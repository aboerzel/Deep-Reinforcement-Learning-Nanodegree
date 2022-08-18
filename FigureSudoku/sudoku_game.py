import math
import random
from tkinter import Tk, Canvas, Frame, BOTH
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog, messagebox
from PIL import Image
from PIL import ImageTk
import tkinter as tk
from tkinter import ttk
import numpy as np


class GridCell:
    def __init__(self, board, row, col, width=80, height=80):
        self.board = board
        self.row = row
        self.col = col

        self.width = width
        self.height = height

        self.x = row * self.width
        self.y = col * self.height

        self.rect = board.create_rectangle(self.y, self.x, self.y + height, self.x + width, outline="black", fill="white")
        self.shape = None

        self.board.tag_bind(self.rect, "<Button-1>", self.clicked)

    def clicked(self, event):
        if self.shape is None:
            #self.board.itemconfig(self.rect, fill='green', outline='red')
            self.shape = self.get_random_shape(color=self.get_random_color())
        else:
            #self.board.itemconfig(self.rect, fill='orange', outline='gray')
            self.board.delete(self.shape)
            self.shape = None

        print(f'Cell {self.row + 1} {self.col + 1} clicked.')

    def create_triangle(self, color='red'):
        r = 15
        ha = self.height - 2 * r
        a = 2 * ha / math.sqrt(3)

        mx = self.width / 2 + self.y
        my = self.height / 2 + self.x

        ax = mx - a / 2
        ay = my + ha / 2

        bx = mx + a / 2
        by = my + ha / 2

        cx = mx
        cy = my - ha / 2

        points = [ax, ay, bx, by, cx, cy]
        shape = self.board.create_polygon(points, smooth=False, fill=color, outline='')
        self.board.tag_bind(shape, "<Button-1>", self.clicked)
        return shape

    def create_circle(self, color='red'):
        r = 15
        x1 = self.y + r
        y1 = self.x + r
        x2 = self.y + self.width - r
        y2 = self.x + self.height - r
        shape = self.board.create_oval(x1, y1, x2, y2, fill=color, outline='')
        self.board.tag_bind(shape, "<Button-1>", self.clicked)
        return shape

    def create_quadrat(self, color='red'):
        r = 15
        x1 = self.y + r
        y1 = self.x + r
        x2 = self.y + self.width - r
        y2 = self.x + self.height - r
        shape = self.board.create_rectangle(x1, y1, x2, y2, fill=color, outline='')
        self.board.tag_bind(shape, "<Button-1>", self.clicked)
        return shape

    def create_hexagon(self, color='red'):
        r = 15
        a = (self.width - (2 * r)) / 2
        ri = math.sqrt(3) * a / 2

        mx = self.width / 2 + self.y
        my = self.height / 2 + self.x

        fx = mx - a
        fy = my

        ax = mx - (a/2)
        ay = my + ri

        bx = mx + (a/2)
        by = my + ri

        cx = mx + a
        cy = my

        dx = mx + (a/2)
        dy = my - ri

        ex = mx - (a/2)
        ey = my - ri

        points = [ax, ay, bx, by, cx, cy, dx, dy, ex, ey, fx, fy]
        shape = self.board.create_polygon(points, smooth=False, fill=color, outline='')
        self.board.tag_bind(shape, "<Button-1>", self.clicked)
        return shape

    def get_random_shape(self, color):
        n = random.randint(0, 4)
        shape = {
            0: self.create_quadrat,
            1: self.create_triangle,
            2: self.create_circle,
            3: self.create_hexagon
        }[n](color=color)
        return shape

    @staticmethod
    def get_random_color():
        n = random.randint(0, 4)
        color = {
            0: 'red',
            1: 'green',
            2: 'yellow',
            3: 'blue'
        }[n]
        return color


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.rows = 4
        self.cols = 4

        self.geometry("320x320")
        self.title('Figure Sudoku')
        self.resizable(False, False)

        # configure the grid
        # self.columnconfigure(0, weight=1)
        # self.columnconfigure(1, weight=3)

        self.grid = np.array([x for x in [[None] * self.rows] * self.cols])

        self.create_board()

    def create_board(self):

        board = Canvas(self)

        width = height = 80

        for row in range(self.rows):
            for col in range(self.cols):
                self.grid[row][col] = GridCell(board, row, col, width=width, height=height)

        board.pack(fill=BOTH, expand=1)


if __name__ == "__main__":
    app = App()
    app.mainloop()

