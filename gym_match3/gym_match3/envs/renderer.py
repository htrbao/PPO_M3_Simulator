import numpy as np
from matplotlib import colors
from gym_match3.envs.game import Board
from gym_match3.envs.constants import GameObject
from IPython import display
import matplotlib
import cv2

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

square_size = 200
class Renderer:
    def __init__(self, n_shapes):
        self.__n_shapes = n_shapes
        self.previousBoard = None
        self.images = []       
        for i in range(1,18):
            self.images.append(cv2.imread(f"/Users/hung/Documents/coding/internVNG/gym-match3/gym_match3/gym_match3/envs/image/{i}.jpg", cv2.IMREAD_UNCHANGED))
        self.square_size = square_size

    def render_board(self, board: Board):
        np_board = board.board
        img = np.zeros((np_board.shape[0]*self.square_size, np_board.shape[1]*self.square_size, 3), dtype=np.uint8)
        for i in range(np_board.shape[0]):
            for j in range(np_board.shape[1]):
                currentObject = int(np_board[i][j])
                small_image = cv2.resize(self.images[currentObject], (self.square_size, self.square_size))
                top_left = (j * self.square_size, i * self.square_size)
                bottom_right = (j * self.square_size + self.square_size, i * self.square_size + self.square_size)
                img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = small_image
        cv2.imshow("board", img)
        cv2.waitKey(1000)
        

