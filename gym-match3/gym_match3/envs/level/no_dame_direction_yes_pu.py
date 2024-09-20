from gym_match3.envs.levels import Level, base_hp
from gym_match3.envs.constants import GameObject
from gym_match3.envs.game import Point, Board, DameMonster, BoxMonster
import numpy as np
import random
import copy

MY_LEVEL= []
# coordinates = [(i,j) for i in range(2,7) for j in range(2,6)]
mons_3 = [(3, 5), (5, 5), (3, 4), (6, 3), (5, 4), (2, 4), (6, 5), (4, 4), (4, 3), (6, 2)]
for i in range(10):
    mons_coor  = mons_3[i]
    board = np.zeros((10, 9))
    board[mons_coor[0]:mons_coor[0]+2, mons_coor[1]:mons_coor[1]+2] = GameObject.monster_dame
    MY_LEVEL.append(
        Level(10, 9, 5, copy.deepcopy(board.tolist()), [
            DameMonster(
                position=Point(mons_coor[0], mons_coor[1]),
                width=2,
                height=2,
                hp=15+base_hp,
                request_masked=[1 if i in random.sample(range(4), random.choice([2, 4])) else 0 for i in range(4)] + [1]
            )
        ])
    )
     
level_3 = MY_LEVEL