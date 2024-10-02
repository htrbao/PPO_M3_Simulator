from gym_match3.envs.levels import Level, base_hp
from gym_match3.envs.constants import GameObject
from gym_match3.envs.game import Point, Board, DameMonster, BoxMonster
import numpy as np
import random
import copy

MY_LEVEL = []
# coordinates = [(i,j) for i in range(1,7) for j in range(2,6)]
mons_5 = [(6, 3), (1, 3), (3, 3), (4, 2), (3, 5), (2, 2), (2, 5), (6, 5), (4, 4), (1, 4)]
for i in range(10):
    mons_coor = mons_5[i]
    board = np.zeros((10, 9))
    mons_w, mons_h = random.choice([(2, 2), (2, 3), (3, 2), (3,3)])
    board[mons_coor[0]:mons_coor[0]+mons_h, mons_coor[1]:mons_coor[1]+mons_w] = GameObject.monster_dame
    MY_LEVEL.append(
        Level(10, 9, 5, copy.deepcopy(board.tolist()), [
            DameMonster(
                position=Point(mons_coor[0], mons_coor[1]),
                width=mons_w,
                height = mons_h,
                hp=20 + base_hp,
                request_masked=[0,0,0,0,1]
            )
        ])
    )
    
    
level_5 = MY_LEVEL
