from gym_match3.envs.constants import GameObject, Level, base_hp
from gym_match3.envs.game import Point, Board, DameMonster, BoxMonster
import numpy as np
import random
import copy


MY_LEVEL = []
# coordinates = [(i,j) for i in range(8) for j in range(7)]
mons_2 = [(6, 2), (0, 6), (3, 3), (5, 0), (4, 1), (4, 5), (1, 2), (3, 2), (3, 4), (2, 0)]
for i in range(10):
    mons_coor = mons_2[i]
    board = np.zeros((10, 9))
    board[mons_coor[0]:mons_coor[0]+2, mons_coor[1]:mons_coor[1]+2] = GameObject.monster_dame
    MY_LEVEL.append(
        Level(10, 9, 5, copy.deepcopy(board.tolist()), [
            DameMonster(
                position=Point(mons_coor[0], mons_coor[1]),
                width=2,
                height=2,
                hp=25+base_hp,
            )
        ])
    )
level_2 = MY_LEVEL