import numpy as np
from collections import namedtuple
import random
import copy

from gym_match3.envs.game import Point, Board, DameMonster, BoxMonster
from gym_match3.envs.constants import GameObject

Level = namedtuple("Level", ["h", "w", "n_shapes", "board", "list_monsters"])


class Match3Levels:

    def __init__(self, levels, immovable_shape=-1, h=None, w=None, n_shapes=None):
        self.__current_lv_idx = 0
        self.__levels = levels
        self.__immovable_shape = immovable_shape
        self.__h = self.__set_dim(h, [lvl.h for lvl in levels])
        self.__w = self.__set_dim(w, [lvl.w for lvl in levels])
        self.__n_shapes = self.__set_dim(n_shapes, [lvl.n_shapes for lvl in levels])

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

    def sample(self):
        """
        :return: board for random level
        """
        level_template = random.sample(self.levels, 1)[0]
        board = self.create_board(level_template)
        return board, level_template.list_monsters
    
    def next(self):
        """
        :return: board for next level
        """
        level_template = self.levels[self.__current_lv_idx]
        board = self.create_board(level_template)
        self.__current_lv_idx = (self.__current_lv_idx + 1) % len(self.levels)
        return board, level_template.list_monsters
    
    def current(self):
        """
        :return: board for next level
        """
        level_template = self.levels[self.__current_lv_idx-1]
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
                raise ValueError('h, w, and n_shapes have to be greater or equal '
                                 'to maximum in levels')
        return d

    def create_board(self, level: Level) -> Board:
        empty_board = np.random.randint(GameObject.color1, self.n_shapes, size=(self.__h, self.__w))
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
        return np.pad(template, [extra_h, extra_w],
                      mode='constant',
                      constant_values=self.__immovable_shape)

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
            even = (extra % 2 == 0)

            if even:
                pad = (extra // 2, extra // 2)
            else:
                pad = (extra // 2 + 1, extra // 2)
        return pad
    
    def get_current_level(self):
        return self.__current_lv_idx-1
    
easy_levels = []
for x in range(0, 8):
    for y in range(0, 7):
        easy_board = [[0 for _ in range(9)] for _ in range(10)]
        for _x in range(x, x + 2):
            for _y in range(y, y + 2):
                easy_board[_x][_y] = GameObject.monster_dame

        easy_levels.append(
            Level(10, 9, 5, copy.deepcopy(easy_board), [
                DameMonster(
                    position=Point(x, y),
                    width=2,
                    height=2,
                    hp=40,
                )])
        )
        easy_levels.append(
            Level(10, 9, 5, copy.deepcopy(easy_board), [
                DameMonster(
                    position=Point(x, y),
                    width=2,
                    height=2,
                    hp=random.randint(20, 25),
                    request_masked=[1, 1, 1, 1, 0]
                )])
        )
        easy_levels.append(
            Level(10, 9, 5, copy.deepcopy(easy_board), [
                DameMonster(
                    position=Point(x, y),
                    width=2,
                    height=2,
                    hp=40,
                    request_masked=[0, 0, 0, 0, 1]
                )])
        )

random.shuffle(easy_levels)

LEVELS = [   
          
    Level(10, 9, 5, [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0],
        [0, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        DameMonster(position=Point(6, 6),
                    relax_interval = 2,
                    setup_interval = 1,
                    width=2,
                    height=2,
                    hp=30,
                    dame=2,
                    have_paper_box=True
                    )
    ]),
    
    *easy_levels,

    Level(10, 9, 5, [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0],
        [0, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        DameMonster(position=Point(6, 6),
                    relax_interval = 2,
                    setup_interval = 1,
                    width=2,
                    height=2,
                    hp=30,
                    dame=2,
                    have_paper_box=True
                    )
    ]),

    Level(10, 9, 5, [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0],
        [0, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        DameMonster(position=Point(6, 6),
                    relax_interval = 2,
                    setup_interval = 1,
                    width=2,
                    height=2,
                    hp=35,
                    dame=2,
                    have_paper_box=True
                    )
    ]),

    Level(10, 9, 5, [
        [-1, -1, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, -1, -1],
    ], [
        DameMonster(position=Point(4, 4),
                    relax_interval = 2,
                    setup_interval = 1,
                    width=2,
                    height=2,
                    hp=30,
                    dame=2,
                    have_paper_box=True
                    )
    ]),

    Level(10, 9, 5, [
        [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
        [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, -1, -1],
        [0, 0, 0, 0, 0, -1, -1, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        DameMonster(position=Point(0, 0),
                    relax_interval = 3,
                    setup_interval = 1,
                    width=2,
                    height=2,
                    hp=35,
                    dame=2,
                    have_paper_box=True
                    )
    ]),

    Level(10, 9, 5, [
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, GameObject.monster_box_bomb, GameObject.monster_box_bomb, 0, 0, 0],
        [-1, 0, 0, 0, GameObject.monster_box_bomb, GameObject.monster_box_bomb, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        BoxMonster(GameObject.monster_box_bomb,
                   position=Point(4, 4),
                   width=2,
                   height=2,
                   hp=30
                   )
    ]),

    Level(10, 9, 5, [
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, GameObject.monster_box_thorny, GameObject.monster_box_thorny, 0, 0, 0],
        [-1, 0, 0, 0, GameObject.monster_box_thorny, GameObject.monster_box_thorny, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        BoxMonster(GameObject.monster_box_thorny,
                   position=Point(4, 4),
                   width=2,
                   height=2,
                   hp=30
                    )
    ]),

    Level(
        10, 9, 5, 
        [
            [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
            [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0],
            [0, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], [
        DameMonster(position=Point(0, 0),
                    width=2,
                    height=2,
                    hp=15
                    ),
        DameMonster(position=Point(7, 6),
                    width=2,
                    height=2,
                    hp=15
                    )
    ]),

    Level(
        10, 9, 5, 
        [
            [0, 0, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame],
            [0, 0, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
            [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
        ], [
        DameMonster(position=Point(0, 7),
                    width=2,
                    height=2,
                    hp=20
                    ),
        DameMonster(position=Point(8, 0),
                    width=2,
                    height=2,
                    hp=15
                    )
    ]),

    Level(10, 9, 5, [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [GameObject.monster_box_box, GameObject.monster_box_box, 0, 0, 0, 0, 0, 0, 0],
        [GameObject.monster_box_box, GameObject.monster_box_box, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
        [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        BoxMonster(GameObject.monster_box_box,
                   position=Point(1, 0),
                   width=2,
                   height=2,
                   hp=15
                   ),
        DameMonster(position=Point(7, 0),
                    width=2,
                    height=2,
                    hp=20
                    )
    ]),

    Level(10, 9, 5, [
        [GameObject.monster_box_both, GameObject.monster_box_both, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame],
        [GameObject.monster_box_both, GameObject.monster_box_both, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        BoxMonster(GameObject.monster_box_both,
                   position=Point(0, 0),
                   width=2,
                   height=2,
                   hp=20
                   ),
        DameMonster(position=Point(0, 7),
                    width=2,
                    height=2,
                    hp=15
                    )
    ]),

    Level(10, 9, 5, [
        [GameObject.monster_box_both, GameObject.monster_box_both, 0, 0, 0, 0, 0, 0, 0],
        [GameObject.monster_box_both, GameObject.monster_box_both, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0],
        [0, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        BoxMonster(GameObject.monster_box_both,
                   position=Point(0, 0),
                   width=2,
                   height=2,
                   hp=20
                   ),
        DameMonster(position=Point(4, 6),
                    relax_interval = 2,
                    setup_interval = 1,
                    width=2,
                    height=2,
                    hp=15,
                    dame=2,
                    have_paper_box=True
                    ),
    ]),

    # Level(10, 9, 5, [
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, GameObject.monster_box_box, GameObject.monster_box_box, 0, 0, 0],
    #     [0, 0, 0, 0, GameObject.monster_box_box, GameObject.monster_box_box, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    # ], [
    #     BoxMonster(box_mons_type=GameObject.monster_box_box,
    #                relax_interval=6,
    #                setup_interval=0, 
    #                position=Point(4, 4),
    #                width=2,
    #                height=2)
    # ]),

    # Level(10, 9, 5, [
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, GameObject.monster_box_bomb, GameObject.monster_box_bomb, 0, 0, 0],
    #     [0, 0, 0, 0, GameObject.monster_box_bomb, GameObject.monster_box_bomb, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    # ], [
    #     BoxMonster(box_mons_type=GameObject.monster_box_bomb,
    #                relax_interval=6,
    #                setup_interval=0,
    #                position=Point(4, 4),
    #                width=2,
    #                height=2)
    # ]),

    # Level(10, 9, 5, [
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, GameObject.monster_box_thorny, GameObject.monster_box_thorny, 0, 0, 0],
    #     [0, 0, 0, 0, GameObject.monster_box_thorny, GameObject.monster_box_thorny, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    # ], [
    #     BoxMonster(box_mons_type=GameObject.monster_box_thorny,
    #                relax_interval=6,
    #                setup_interval=0, 
    #                position=Point(4, 4),
    #                width=2,
    #                height=2)
    # ]),
]

