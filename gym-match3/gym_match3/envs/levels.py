import numpy as np
from collections import namedtuple
import random
import copy

from gym_match3.envs.game import Point, Board, DameMonster, BoxMonster
from gym_match3.envs.constants import GameObject, Level, base_hp
from gym_match3.envs.level import ALL_LEVELS
from enum import Enum
import os
import importlib

dict_levels = {}


class Match3Levels:

    def __init__(self, levels, immovable_shape=-1, h=None, w=None, n_shapes=None):
        self.__current_lv_idx = 0
        self.__levels = levels
        self.__immovable_shape = immovable_shape
        self.__h = self.__set_dim(h, [lvl.h for lvl in levels])
        self.__w = self.__set_dim(w, [lvl.w for lvl in levels])
        self.__n_shapes = self.__set_dim(n_shapes, [lvl.n_shapes for lvl in levels])
        # self.__choosing_ratio = np.full(levels.shape, 1. / len(levels))

    @property
    def h(self):
        return self.__h

    @property
    def w(self):
        return self.__w

    @property
    def n_shapes(self):
        return self.__n_shapes

    @property
    def levels(self):
        return self.__levels
    
    @property
    def current_level(self):
        return self.__current_lv_idx % len(self.levels)

    def sample(self):
        """
        :return: board for random level
        """
        level_template = random.sample(self.levels, 1)[0]
        board = self.create_board(level_template)
        return board, level_template.list_monsters

    def next(self, is_win: bool):
        """
        :return: board for random level
        """
        # if is_win is not None:
        #     self.__num_plays[self.__current_lv_idx] += 1
        #     self.__num_wins[self.__current_lv_idx] += int(is_win)
        # self.__current_lv_idx = random.choices(
        #     [i for i in range(len(self.levels))],
        #     weights=[
        #         (
        #             2 - self.__num_wins[i] / self.__num_plays[i]
        #             if self.__num_plays[i] != 0
        #             else 2
        #         )
        #         for i in range(len(self.levels))
        #     ],
        #     k=1
        # )[0]

        self.__current_lv_idx = (self.__current_lv_idx + 1) % len(self.levels)

        level_template = self.levels[self.__current_lv_idx]
        board = self.create_board(level_template)
        return board, level_template.list_monsters

    @staticmethod
    def __set_dim(d, ds):
        """
        :param d: int or None, size of dimenstion
        :param ds: iterable, dim's sizes of levels
        :return: int, dim's size
        """
        max_ = max(ds)
        if d is None:
            d = max_
        else:
            if d < max_:
                raise ValueError(
                    "h, w, and n_shapes have to be greater or equal "
                    "to maximum in levels"
                )
        return d

    # def __calc_ratio(self, num_wins, num_plays):
    #     return num_wins / num_plays

    # def apply_ratio(self, num_wins, num_plays):
    #     self.__choosing_ratio = self.__calc_ratio(num_wins, num_plays)

    def create_board(self, level: Level) -> Board:
        empty_board = np.random.randint(
            GameObject.color1, self.n_shapes, size=(self.__h, self.__w)
        )
        board_array = self.__put_immovable(empty_board, level)
        board_array = self.__put_monster(empty_board, level)
        board = Board(self.__h, self.__w, level.n_shapes, self.__immovable_shape)
        board.set_board(board_array)
        return board

    def __put_immovable(self, board, level):
        template = np.array(level.board)
        expanded_template = self.__expand_template(template)
        board[expanded_template == self.__immovable_shape] = -1
        return board

    def __put_monster(self, board, level):
        template = np.array(level.board)
        expanded_template = self.__expand_template(template)
        for monster in GameObject.monsters:
            board[expanded_template == monster] = monster
        return board

    def __expand_template(self, template):
        """
        pad template of a board to maximum size in levels by immovable_shapes
        :param template: board for level
        :return:
        """
        template_h, template_w = template.shape
        extra_h, extra_w = self.__calc_extra_dims(template_h, template_w)
        return np.pad(
            template,
            [extra_h, extra_w],
            mode="constant",
            constant_values=self.__immovable_shape,
        )

    def __calc_extra_dims(self, h, w):
        pad_h = self.__calc_padding(h, self.h)
        pad_w = self.__calc_padding(w, self.w)
        return pad_h, pad_w

    @staticmethod
    def __calc_padding(size, req_size):
        """
        calculate padding size for dimension
        :param size: int, size of level's dimension
        :param req_size: int, required size of dimension
        :return: tuple of ints with pad width
        """
        assert req_size >= size
        if req_size == size:
            pad = (0, 0)

        else:
            extra = req_size - size
            even = extra % 2 == 0

            if even:
                pad = (extra // 2, extra // 2)
            else:
                pad = (extra // 2 + 1, extra // 2)

        return pad


easy_levels = []
shield_one_mon_level = []
paper_box_mon_level = []
for x in range(0, 9):
    for y in range(0, 8):
        easy_board = [[0 for _ in range(9)] for _ in range(10)]
        for _x in range(x, x + 2):
            for _y in range(y, y + 2):
                easy_board[_x][_y] = GameObject.monster_dame
        if x == 0 or y == 0 or x == 8 or y == 7 or x % 3 == 0 or y % 3 == 0:
            easy_levels.append(
                Level(
                    10,
                    9,
                    4,
                    copy.deepcopy(easy_board),
                    [
                        DameMonster(
                            position=Point(x, y),
                            width=2,
                            height=2,
                            hp=(90 + (y - 9) if (x == 0 or y == 0 or x == 8 or y == 7) else 100) + base_hp,
                        )
                    ],
                )
            )
        if x == 0 or y == 0 or x == 8 or y == 7 or x % 2 == 0 or y % 2 == 0:
            shield_one_mon_level.append(
                Level(
                    10,
                    9,
                    4,
                    copy.deepcopy(easy_board),
                    [
                        DameMonster(
                            position=Point(x, y),
                            width=2,
                            height=2,
                            hp=(35 if (x == 0 or y == 0 or x == 8 or y == 7) else 45) + base_hp,
                            request_masked=[1, 1, 1, 1, 0],
                        )
                    ],
                )
            )
            shield_one_mon_level.append(
                Level(
                    10,
                    9,
                    4,
                    copy.deepcopy(easy_board),
                    [
                        DameMonster(
                            position=Point(x, y),
                            width=2,
                            height=2,
                            hp=(60 if (x == 0 or y == 0 or x == 8 or y == 7) else 60) + base_hp,
                            request_masked=[0, 0, 0, 0, 1],
                        )
                    ],
                )
            )
            paper_box_mon_level.append(
                Level(
                    10,
                    9,
                    4,
                    copy.deepcopy(easy_board),
                    [
                        DameMonster(
                            position=Point(x, y),
                            width=2,
                            height=2,
                            hp=(35 if (x == 0 or y == 0 or x == 8 or y == 7) else 45) + base_hp,
                            have_paper_box=True,
                            setup_interval= 6,
                            relax_interval= 2,
                        )
                    ],
                )
            )
# print("len easy", len(easy_levels))
# print("len shield", len(shield_one_mon_level))
# print("len paper_box", len(paper_box_mon_level))
LEVELS = [
    *easy_levels,
    *shield_one_mon_level,
    *paper_box_mon_level,
    Level(
        10,
        9,
        5,
        [
            [
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
            ],
            [
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
            ],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            DameMonster(
                position=Point(0, 0),
                relax_interval=2,
                setup_interval=1,
                width=9,
                height=2,
                hp=65,
                dame=3,
            )
        ],
    ),
    Level(
        10,
        9,
        5,
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
            ],
            [
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
                GameObject.monster_dame,
            ],
        ],
        [
            DameMonster(
                position=Point(8, 0),
                relax_interval=2,
                setup_interval=1,
                width=9,
                height=2,
                hp=65,
                dame=3,
            )
        ],
    ),
    Level(
        10,
        9,
        5,
        [
            [GameObject.monster_dame, GameObject.monster_dame, -1, 0, 0, 0, 0, 0, 0],
            [GameObject.monster_dame, GameObject.monster_dame, -1, 0, 0, 0, 0, 0, 0],
            [GameObject.monster_dame, GameObject.monster_dame, -1, 0, 0, 0, 0, 0, 0],
            [GameObject.monster_dame, GameObject.monster_dame, -1, 0, 0, 0, 0, 0, 0],
            [GameObject.monster_dame, GameObject.monster_dame, -1, 0, 0, 0, 0, 0, 0],
            [GameObject.monster_dame, GameObject.monster_dame, -1, 0, 0, 0, 0, 0, 0],
            [GameObject.monster_dame, GameObject.monster_dame, -1, 0, 0, 0, 0, 0, 0],
            [GameObject.monster_dame, GameObject.monster_dame, -1, 0, 0, 0, 0, 0, 0],
            [GameObject.monster_dame, GameObject.monster_dame, -1, 0, 0, 0, 0, 0, 0],
            [GameObject.monster_dame, GameObject.monster_dame, -1, 0, 0, 0, 0, 0, 0],
        ],
        [
            DameMonster(
                position=Point(0, 0),
                relax_interval=2,
                setup_interval=1,
                width=2,
                height=10,
                hp=65,
                dame=3,
            )
        ],
    ),
    Level(
        10,
        9,
        5,
        [
            [0, 0, 0, 0, 0, 0, -1, GameObject.monster_dame, GameObject.monster_dame],
            [0, 0, 0, 0, 0, 0, -1, GameObject.monster_dame, GameObject.monster_dame],
            [0, 0, 0, 0, 0, 0, -1, GameObject.monster_dame, GameObject.monster_dame],
            [0, 0, 0, 0, 0, 0, -1, GameObject.monster_dame, GameObject.monster_dame],
            [0, 0, 0, 0, 0, 0, -1, GameObject.monster_dame, GameObject.monster_dame],
            [0, 0, 0, 0, 0, 0, -1, GameObject.monster_dame, GameObject.monster_dame],
            [0, 0, 0, 0, 0, 0, -1, GameObject.monster_dame, GameObject.monster_dame],
            [0, 0, 0, 0, 0, 0, -1, GameObject.monster_dame, GameObject.monster_dame],
            [0, 0, 0, 0, 0, 0, -1, GameObject.monster_dame, GameObject.monster_dame],
            [0, 0, 0, 0, 0, 0, -1, GameObject.monster_dame, GameObject.monster_dame],
        ],
        [
            DameMonster(
                position=Point(0, 7),
                relax_interval=2,
                setup_interval=1,
                width=2,
                height=10,
                hp=65,
                dame=3,
            )
        ],
    ),
    *ALL_LEVELS,
]


HEIGHT = 10
WIDTH = 9
min_height_mons = 2
max_height_mons = 3
max_step = 90

class MonsterType(Enum):
    DMG_MONSTER = 0
    BLOCKER_MONSTER = 1
    PAPER_BOX_MONSTER = 2
    def __str__(self):
        return self.name


def condition_check_tiles(x, y, request_masked, paper_box, hp, height_mon):
    # if monster is on side of board, this function return num tiles = 3
    # if monster is not near side of board, num tiles is 4 or 5
    # if monster sum of request_masked <= 2 or paper_box is True or hp is smaller < 50, num tiles is 3
    # else random num tiles 4 and 5  
    
    # If monster is on a side of the board, return num_tiles = 3
    # if (x == 0 or y == 0 or x == WIDTH - height_mon or y == HEIGHT - height_mon  or hp > 60) or sum(request_masked) <= 2 or paper_box:
    #     return 3
    return np.random.choice([4, 5], p=[0.4, 0.6])

def generate_request_masked(y, x, hp):
    # default request masked is boolean array [1, 1, 1, 1, 1] meaning monster take dmg from left, right, top, bottom, inside
    # using this function to generate request masked based on the given position and its position
    # if monster is on side of board, request masked will have at least 3 direction to take dmg example [0, 0, 1, 1, 1]
    # if monster is not near side of board, request masked will at least 1 or 2 direction to take dmg example [1, 0, 0, 0, 1]
    
    # Default request mask is [1, 1, 1, 1, 1] (left, right, top, bottom, inside)
    request_masked = [1, 1, 1, 1, 1]

    if x <= WIDTH // 2:
        request_masked[1] = 0
    else:
        request_masked[0] = 0

    if y <= HEIGHT // 2:
        request_masked[3] = 0
    else:
        request_masked[2] = 0
    
    if x > 0 and x < WIDTH - 1 and y > 0 or y < HEIGHT - 1:
        for _ in range(2):
            possible_directions = [i for i, v in enumerate(request_masked) if v == 1]
            if possible_directions and sum(request_masked) >= 3:
                disable_direction = random.choice(possible_directions)
                request_masked[disable_direction] = 0
            
    return request_masked

def generate_max_mons_hp(x, y, type, height_mon):
    # Default HP values for different monster types
    default_hp = {MonsterType.DMG_MONSTER: 80, MonsterType.BLOCKER_MONSTER: 80, MonsterType.PAPER_BOX_MONSTER: 80}
    
    # Get the base HP for the given monster type
    hp = default_hp.get(type, 40)  # Default to 40 if type is not recognized

    # Decrease HP if the monster is near the side of the board
    if x == 0 or y == 0 or x == WIDTH - height_mon or y == HEIGHT - height_mon:
        hp -= (5 + 2 * type.value)

    # Increase HP if the monster's area (height x WIDTH) is greater than 4
    if (height_mon ** 2) > 4:
        hp += (5 - 1 * type.value) 

    return hp

def merge_mutliple_level(levels1, levels2):
    merged_levels = []
    for i, (level1, kwargs1) in enumerate(levels1):
        for j, (level2, kwargs2) in enumerate(levels2):
            if i >= j:
                continue  

            # Extract monster positions and dimensions
            monster1 = level1.list_monsters[0]
            monster2 = level2.list_monsters[0]

            new_board = np.maximum(level1.board, level2.board)
            monster1_coord = kwargs1["position"].get_coord()
            monster2_coord = kwargs2["position"].get_coord()

            if ((
                monster1_coord[1] + kwargs1["width"] <= monster2_coord[1] or
                monster2_coord[1] + kwargs2["width"] <= monster1_coord[1]
                ) 
                or 
                (
                monster1_coord[0] + kwargs1["height"] <= monster2_coord[0] or
                monster2_coord[0] + kwargs2["height"] <= monster1_coord[0]
                )
                ):

                # Create a new board by combining the two boards
                
                # Create a new level with the two monsters
                new_level = Level(
                    h=10,
                    w=9,
                    n_shapes=min(level1.n_shapes, level2.n_shapes),
                    board=new_board,
                    list_monsters=[monster1, monster2]
                )
                
                merged_levels.append(new_level)
    return merged_levels

LOC_LEVELS = {
    MonsterType.DMG_MONSTER: [],
    MonsterType.BLOCKER_MONSTER: [],
    MonsterType.PAPER_BOX_MONSTER: []
}
for y in range(0, 9, 2):
    for x in range(0, 8):
        for height_mon in range(min_height_mons, max_height_mons+1):
            # print(height_mon + y, x + height_mon)
            if not (height_mon + y < HEIGHT and x + height_mon < WIDTH):
                continue
            easy_board = np.zeros((HEIGHT, WIDTH), dtype=int)
            easy_board[y: y+height_mon,x: x+height_mon] = GameObject.monster_dame
            for type_monster in list(MonsterType):
                hp = generate_max_mons_hp(x, y, type_monster, height_mon)
                monster_kwargs = {
                    'position': Point(y, x),
                    'width': height_mon,
                    'height': height_mon,
                    'hp': hp,
                    'dame': 0
                }
                
                if type_monster == MonsterType.BLOCKER_MONSTER:
                    monster_kwargs['request_masked'] = generate_request_masked(y, x, hp)
                elif type_monster == MonsterType.PAPER_BOX_MONSTER:
                    monster_kwargs['have_paper_box'] = True
                    monster_kwargs['relax_interval'] = 5
                    monster_kwargs['setup_interval'] = 2
                num_tiles = condition_check_tiles(x, y, 
                                                monster_kwargs.get("request_masked", [1, 1, 1, 1, 1]), 
                                                monster_kwargs.get("have_paper_box", False), 
                                                hp, 
                                                height_mon)
                monster = DameMonster(
                                **monster_kwargs
                            )
                LOC_LEVELS[type_monster].append(
                    (Level(
                        10,
                        9,
                        num_tiles,
                        easy_board.copy(),
                        [
                            monster
                        ],
                    ), monster_kwargs)
                )

MULTI_LOC_LEVELS = {
    MonsterType.DMG_MONSTER: merge_mutliple_level(LOC_LEVELS[MonsterType.DMG_MONSTER], LOC_LEVELS[MonsterType.DMG_MONSTER]),
    MonsterType.BLOCKER_MONSTER: merge_mutliple_level(LOC_LEVELS[MonsterType.BLOCKER_MONSTER], LOC_LEVELS[MonsterType.BLOCKER_MONSTER]),
    MonsterType.PAPER_BOX_MONSTER: merge_mutliple_level(LOC_LEVELS[MonsterType.PAPER_BOX_MONSTER], LOC_LEVELS[MonsterType.PAPER_BOX_MONSTER]),
}

MULTI_LOC_LEVELS[MonsterType.DMG_MONSTER] = [l[0] for l in LOC_LEVELS[MonsterType.DMG_MONSTER]] + MULTI_LOC_LEVELS[MonsterType.DMG_MONSTER]
MULTI_LOC_LEVELS[MonsterType.BLOCKER_MONSTER] = [l[0] for l in LOC_LEVELS[MonsterType.BLOCKER_MONSTER]] + MULTI_LOC_LEVELS[MonsterType.BLOCKER_MONSTER]
MULTI_LOC_LEVELS[MonsterType.PAPER_BOX_MONSTER] = [l[0] for l in LOC_LEVELS[MonsterType.PAPER_BOX_MONSTER]] + MULTI_LOC_LEVELS[MonsterType.PAPER_BOX_MONSTER]


LARGE_LOC_LEVELS = []
for dmg_level, block_level, paper_level in zip(LOC_LEVELS[MonsterType.DMG_MONSTER], LOC_LEVELS[MonsterType.BLOCKER_MONSTER], LOC_LEVELS[MonsterType.PAPER_BOX_MONSTER]):
    LARGE_LOC_LEVELS.extend([dmg_level, block_level, paper_level])

LARGE_MULTI_LOC_LEVELS = merge_mutliple_level(LARGE_LOC_LEVELS, LARGE_LOC_LEVELS)
LARGE_MULTI_LOC_LEVELS = [l[0] for l in LARGE_LOC_LEVELS] + LARGE_MULTI_LOC_LEVELS