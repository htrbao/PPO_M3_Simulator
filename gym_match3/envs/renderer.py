import numpy as np
from matplotlib import colors
from gym_match3.envs.game import Board
from gym_match3.envs.constants import GameObject
from IPython import display
import matplotlib
import cv2

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

square_size = 400
class Renderer:
    def __init__(self, n_shapes):
        self.__n_shapes = n_shapes
        self.previousBoard = None
        self.images = []       
        for i in range(0,17):
            self.images.append(cv2.imread(f"./gym_match3/envs/image/{i}.jpg", cv2.IMREAD_UNCHANGED))
        self.square_size = square_size

    def render_board(self, board: Board, tiles=None):
        np_board = board.board
        img = np.zeros((np_board.shape[0]*self.square_size, np_board.shape[1]*self.square_size, 3), dtype=np.uint8)
        actions = 0
        for i in range(np_board.shape[0]):
            for j in range(np_board.shape[1]):
                currentObject = int(np_board[i][j])
                small_image = cv2.resize(self.images[currentObject], (self.square_size, self.square_size))
                top_left = (j * self.square_size, i * self.square_size)
                bottom_right = (j * self.square_size + self.square_size, i * self.square_size + self.square_size)
                img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = small_image
                cv2.putText(img, str(f'{i},{j}'), (j * self.square_size + self.square_size//2- 40, i * self.square_size + self.square_size//2 +8), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
                actions += 1
                # if j > 0 and j < np_board.shape[1] - 1:
                #     cv2.putText(img, str(f'{actions}'), (j * self.square_size + self.square_size//2, i * self.square_size + self.square_size//2 +10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,0), 3, cv2.LINE_AA)
                
        cv2.imshow("board", img)
        cv2.waitKey(1000)
        print(f"Selected action: {tiles}")
        x1,y1,x2,y2 = tiles['x1'], tiles['y1'], tiles['x2'], tiles['y2']
        first = int(np_board[x1][y1])
        second = int(np_board[x2][y2])
        imgFirst = cv2.resize(self.images[first], (self.square_size, self.square_size))
        imgSecond = cv2.resize(self.images[second], (self.square_size, self.square_size))
        vertical = x1 != x2
        img[x1 * self.square_size:x1 * self.square_size + self.square_size, y1 * self.square_size:y1 * self.square_size + self.square_size] = (255,255,255)
        img[x2 * self.square_size:x2 * self.square_size + self.square_size, y2 * self.square_size:y2 * self.square_size + self.square_size] = (255,255,255)
        for i in range(0,self.square_size+1,40):
            if vertical:
                top_left = (y1 * self.square_size, x1 * self.square_size + i)
                bottom_right = (y1 * self.square_size + self.square_size, x1 * self.square_size + self.square_size + i)
                img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = imgFirst
                top_left2 = (y2 * self.square_size, x2 * self.square_size - i)
                bottom_right2 = (y2 * self.square_size + self.square_size, x2 * self.square_size + self.square_size - i)
                img[top_left2[1]:bottom_right2[1], top_left2[0]:bottom_right2[0]] = imgSecond   
            else:
                top_left = (y1 * self.square_size + i, x1 * self.square_size)
                bottom_right = (y1 * self.square_size + self.square_size + i, x1 * self.square_size + self.square_size)
                img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = imgFirst
                top_left2 = (y2 * self.square_size - i, x2 * self.square_size)
                bottom_right2 = (y2 * self.square_size + self.square_size - i, x2 * self.square_size + self.square_size)
                img[top_left2[1]:bottom_right2[1], top_left2[0]:bottom_right2[0]] = imgSecond
            cv2.imshow("board", img)
            cv2.waitKey(1)
            img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = (255,255,255)
            img[top_left2[1]:bottom_right2[1], top_left2[0]:bottom_right2[0]] = (255,255,255)
        cv2.waitKey(0)