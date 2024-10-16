import copy
from typing import Union
from itertools import product
from functools import wraps
from abc import ABC, abstractmethod
import numpy as np
import random

import traceback
import threading

from gym_match3.envs.constants import GameAction, GameObject, mask_immov_mask, need_to_match
from gym_match3.envs.functions import is_valid_point
from gym_match3.envs.cython_functions import cfunctions


class OutOfBoardError(IndexError):
    pass


class ImmovableShapeError(ValueError):
    pass


class AbstractPoint(ABC):

    @abstractmethod
    def get_coord(self) -> tuple:
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass


class Point(AbstractPoint):
    """pointer to coordinates on the board"""

    def __init__(self, row, col):
        self.__row = row
        self.__col = col

    def set_coord(self, row, col):
        self.__row = row
        self.__col = col

    def get_coord(self):
        return self.__row, self.__col

    def euclidean_distance(self, another):
        return np.linalg.norm(np.array(self.get_coord()) - np.array(another.get_coord()))

    def __add__(self, other):
        row1, col1 = self.get_coord()
        row2, col2 = other.get_coord()
        return Point(row1 + row2, col1 + col2)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, constant):
        row, col = self.get_coord()
        return Point(row * constant, col * constant)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return -1 * other + self

    def __eq__(self, other):
        return self.get_coord() == other.get_coord()

    def __hash__(self):
        return hash(self.get_coord())

    def __str__(self):
        return str(self.get_coord())

    def __repr__(self):
        return self.__str__()


class Cell(Point):
    def __init__(self, shape, row, col):
        self.__shape = shape
        super().__init__(row, col)

    def set_cell(self, shape, row, col):
        self.__shape = shape
        self.set_coord(row, col)

    @property
    def shape(self):
        return self.__shape

    @property
    def point(self):
        return Point(*self.get_coord())

    def __eq__(self, other):
        if isinstance(other, Point):
            return super().__eq__(other)
        else:
            eq_shape = self.shape == other.shape
            eq_points = super().__eq__(other)
            return eq_shape and eq_points

    def __hash__(self):
        return hash((self.shape, self.get_coord()))


class AbstractBoard(ABC):

    @property
    @abstractmethod
    def board(self):
        pass

    @property
    @abstractmethod
    def board_size(self):
        pass

    @property
    @abstractmethod
    def n_shapes(self):
        pass

    @abstractmethod
    def swap(self, point1: Point, point2: Point):
        pass

    @abstractmethod
    def set_board(self, board: np.ndarray):
        pass

    @abstractmethod
    def move(self, point: Point, direction: Point):
        pass

    @abstractmethod
    def shuffle(self, random_state=None):
        pass

    @abstractmethod
    def get_shape(self, point: Point):
        pass

    @abstractmethod
    def delete(self, points):
        pass

    @abstractmethod
    def get_line(self, ind):
        pass

    @abstractmethod
    def put_line(self, ind, line):
        pass

    @abstractmethod
    def put_mask(self, mask, shapes):
        pass


def check_availability_dec(func):
    @wraps(func)
    def wrapped(self, *args):
        self._check_availability(*args)
        return func(self, *args)

    return wrapped


class Board(AbstractBoard):
    """board for match3 game"""

    def __init__(self, rows, columns, n_shapes, immovable_shape=-1, random_state=None):
        self.__rows = rows
        self.__columns = columns
        self.__n_shapes = n_shapes
        self.__immovable_shape = immovable_shape
        self.__board = None  # np.ndarray
        self.power_points = set()
        self.__random_state = random_state
        np.random.seed(self.__random_state)

        if 0 <= immovable_shape < n_shapes:
            raise ValueError("Immovable shape has to be less or greater than n_shapes")

    def __getitem__(self, indx: Point):
        # self.__check_board()
        self.__validate_points(indx)
        if isinstance(indx, Point):
            return self.board.__getitem__(indx.get_coord())
        else:
            raise ValueError("Only Point class supported for getting shapes")

    def __setitem__(self, value, indx: Point):
        self.__check_board()
        # print(indx)
        self.__validate_points(indx)
        if isinstance(indx, Point):
            self.__board[indx.get_coord()] = value
        else:
            raise ValueError("Only Point class supported for setting shapes")

    def __str__(self):
        if isinstance(self.board, np.ndarray):
            return str(self.board)
        return self.board.board

    @property
    def immovable_shape(self):
        return self.__immovable_shape

    @property
    def board(self):
        self.__check_board()
        return self.__board

    @property
    def board_size(self):
        return self.__rows, self.__columns

    def set_board(self, board: np.ndarray):
        self.__validate_board(board)
        self.__board = board.astype(float)
        self.__rows, self.__columns = board.shape

    def shuffle(self, random_state=None):
        moveable_mask = self.board != self.immovable_shape
        board_ravel = self.board[moveable_mask]
        np.random.shuffle(board_ravel)
        self.put_mask(moveable_mask, board_ravel)

    def __check_board(self):
        if not self.__is_board_exist():
            raise ValueError("Board is not created")

    @property
    def n_shapes(self):
        return self.__n_shapes

    @check_availability_dec
    def swap(self, point1: Point, point2: Point):
        point1_shape = self.get_shape(point1)
        point2_shape = self.get_shape(point2)
        self.put_shape(point2, point1_shape)
        self.put_shape(point1, point2_shape)

    def put_shape(self, shape, point: Point):
        self[point] = shape

        # self.change_shape(shape, point)

    def determine_power_points(self):
        self.power_points.clear()
        for i, j in product(range(self.__rows), range(self.__columns)):
            if self.__board.__getitem__((i, j)) in GameObject.set_powers_shape:
                self.power_points.add((i, j))

    def move(self, point: Point, direction: Point):
        self._check_availability(point)
        new_point = point + direction
        self.swap(point, new_point)

    def __is_board_exist(self):
        existence = self.__board is not None
        return existence

    def __validate_board(self, board: np.ndarray):
        # self.__validate_max_shape(board) # No check here because of multi tile
        self.__validate_board_size(board)

    def __validate_board_size(self, board: np.ndarray):
        provided_board_shape = board.shape
        right_board_shape = self.board_size
        correct_shape = provided_board_shape == right_board_shape
        if not correct_shape:
            raise ValueError(
                "Incorrect board shape: "
                f"{provided_board_shape} vs {right_board_shape}"
            )

    def __validate_max_shape(self, board: np.ndarray):
        if np.all(np.isnan(board)):
            return
        provided_max_shape = np.nanmax(board)

        right_max_shape = self.n_shapes
        if provided_max_shape > right_max_shape:
            raise ValueError(
                "Incorrect shapes of the board: "
                f"{provided_max_shape} vs {right_max_shape}"
            )

    def get_shape(self, point: Point):
        return self[point]

    def get_valid_shape(self, row, col):
        return self.__board.__getitem__((row, col))

    def __validate_points(self, *args):
        board_rows, board_cols = self.board_size

        for point in args:
            row, col = point.get_coord()
            is_valid = is_valid_point(row, col, board_rows, board_cols)
            if not is_valid:
                raise OutOfBoardError(f"Invalid point: {point.get_coord()}")

    def __is_valid_point(self, point: Point):
        row, col = point.get_coord()
        board_rows, board_cols = self.board_size
        correct_row = ((row + 1) <= board_rows) and (row >= 0)
        correct_col = ((col + 1) <= board_cols) and (col >= 0)
        return correct_row and correct_col

    def is_valid_point(self, row, col):
        board_rows, board_cols = self.board_size
        correct_row = ((row + 1) <= board_rows) and (row >= 0)
        correct_col = ((col + 1) <= board_cols) and (col >= 0)
        return correct_row and correct_col

    def _check_availability(self, *args):
        for p in args:
            shape = self.get_shape(p)
            if shape == GameObject.immovable_shape:
                raise ImmovableShapeError

    def delete(self, points: set[Point], allow_delete_monsters: bool = False):
        if allow_delete_monsters:
            coordinates = tuple(np.array([i.get_coord() for i in points]).T.tolist())
        else:
            coordinates = tuple(
                np.array(
                    [
                        i.get_coord()
                        for i in points
                        if self.get_valid_shape(*i.get_coord()) not in GameObject.set_unmovable_shape
                    ]
                ).T.tolist()
            )
        self.__board[coordinates] = np.nan
        # print("DELETE: ", self.__board)
        return self

    def get_line(self, ind, axis=1):
        return np.take(self.board, ind, axis=axis)

    def get_monster(self):
        return [
            Point(i, j)
            for i, j in product(range(self.board_size[0]), range(self.board_size[1]))
            if self.get_shape(Point(i, j)) in GameObject.set_monsters_shape
        ]

    def put_line(self, ind, line: np.ndarray):
        # TODO: create board with putting lines on arbitrary axis
        self.__validate_line(ind, line)
        # self.__validate_max_shape(line)
        self.__board[:, ind] = line
        return self

    def put_mask(self, mask, shapes):
        self.__validate_mask(mask)
        # self.__validate_max_shape(shapes)
        self.__board[mask] = shapes
        return self

    def __validate_mask(self, mask):
        if np.any(self.board[mask] == self.immovable_shape):
            raise ImmovableShapeError

    def __validate_line(self, ind, line):
        immove_mask = mask_immov_mask(self.board[:, ind], self.immovable_shape)
        new_immove_mask = mask_immov_mask(np.array(line), self.immovable_shape)
        # print(immove_mask)
        # print(new_immove_mask)
        if not np.array_equal(immove_mask, new_immove_mask):
            raise ImmovableShapeError


class RandomBoard(Board):

    def set_random_board(self, random_state=None):
        board_size = self.board_size
        board = np.random.randint(
            low=GameObject.color1, high=self.n_shapes + 1, size=board_size
        )
        self.set_board(board)
        return self


class CustomBoard(Board):

    def __init__(self, board: np.ndarray, n_shapes: int):
        columns, rows = board.shape
        super().__init__(columns, rows, n_shapes)
        self.set_board(board)


class AbstractSearcher(ABC):
    def __init__(self, board_ndim):
        self.__directions = self.__get_directions(board_ndim)
        self.__disco_directions = self.__get_disco_directions(board_ndim)
        self.__bomb_directions = self.__get_bomb_directions(board_ndim)
        self.__missile_directions = self.__get_missile_directions(board_ndim)
        self.__plane_directions = self.__get_plane_directions(board_ndim)
        self.__power_up_cls = (
            [GameObject.power_disco] * len(self.__disco_directions)
            + [GameObject.power_bomb] * len(self.__bomb_directions)
            + [GameObject.power_missile_h, GameObject.power_missile_v]
            + [GameObject.power_plane] * len(self.__plane_directions)
            + [-1] * len(self.__directions)
        )

        # All possible directions can occur in game
        self.__full_directions = (
                self.__disco_directions
                + self.__bomb_directions
                + self.__missile_directions
                + self.__plane_directions
                + self.__directions
        )

    @staticmethod
    def __get_directions(board_ndim):
        directions = [
            [[0 for _ in range(board_ndim)] for _ in range(2)]
            for _ in range(board_ndim)
        ]
        for ind in range(board_ndim):
            directions[ind][0][ind] = 1
            directions[ind][1][ind] = -1
        return directions

    @staticmethod
    def __get_disco_directions(board_ndim):
        directions = [
            [[0 for _ in range(board_ndim)] for _ in range(4)]
            for _ in range(board_ndim)
        ]
        for ind in range(board_ndim):
            directions[ind][0][ind] = -2
            directions[ind][1][ind] = -1
            directions[ind][2][ind] = 1
            directions[ind][3][ind] = 2
        return directions

    @staticmethod
    def __get_plane_directions(board_ndim):
        directions = [[[0, 1], [1, 0], [1, 1]]]
        return directions

    @staticmethod
    def __get_bomb_directions(board_ndim):
        directions_T = [
            [[0 for _ in range(board_ndim)] for _ in range(4)] for _ in range(5)
        ]
        for ind in range(len(directions_T)):
            directions_T[ind][0][0] = -1
            directions_T[ind][1][0] = 1
            directions_T[ind][2][1] = -1
            directions_T[ind][3][1] = 1
        for ind in range(1, 5):
            coeff = int(ind > 2) * 2
            directions_T[ind][0 + coeff][ind < 3] = -1 + (ind % 2) * 2
            directions_T[ind][1 + coeff][ind < 3] = -1 + (ind % 2) * 2

        directions_L = [
            [[0 for _ in range(board_ndim)] for _ in range(4)] for _ in range(4)
        ]
        for ind in range(4):
            coeff = ind % 2 * 2
            directions_L[ind][0 + coeff][ind % 2] = -2 if 0 < ind and ind < 3 else 2
            directions_L[ind][1 + coeff][ind % 2] = -1 if 0 < ind and ind < 3 else 1

            directions_L[(ind + 1) % 4][0 + coeff][ind % 2] = (
                -2 if 0 < ind and ind < 3 else 2
            )
            directions_L[(ind + 1) % 4][1 + coeff][ind % 2] = (
                -1 if 0 < ind and ind < 3 else 1
            )

        return directions_T + directions_L

    @staticmethod
    def __get_missile_directions(board_ndim):
        directions = [
            [[0 for _ in range(board_ndim)] for _ in range(3)]
            for _ in range(board_ndim)
        ]
        for ind in range(board_ndim):
            directions[ind][0][ind] = -2
            directions[ind][1][ind] = -1
            directions[ind][2][ind] = 1
        return directions

    def get_power_up_type(self, ind):
        return self.__power_up_cls[ind]

    @property
    def full_directions(self):
        return self.__full_directions

    @property
    def normal_directions(self):
        return self.__directions

    @property
    def plane_directions(self):
        return self.__plane_directions

    @staticmethod
    def generate_movable_points(board: Board, focus_range=None):
        rows, cols = board.board_size

        if not focus_range:
            loop_range = product(range(rows), range(cols))
        else:
            max_row, start_col, end_col = focus_range
            loop_range = product(range(0, min(max_row + 1, rows)), range(max(start_col, 0), min(end_col + 1, cols)))

        board_contain_shapes = board.board

        for i, j in loop_range:
            shape = board_contain_shapes.__getitem__((i, j))
            if shape != board.immovable_shape and need_to_match(shape):
                yield Point(i, j)

    @staticmethod
    def generate_movable_point(rows, cols, board_contain_shapes, focus_range=None,
                               set_shapes=GameObject.set_movable_shape):
        if not focus_range:
            loop_range = product(range(rows), range(cols))
        else:
            max_row, start_col, end_col = focus_range
            loop_range = product(range(0, min(max_row + 1, rows)), range(max(start_col, 0), min(end_col + 1, cols)))

        return (Point(i, j) for i, j in loop_range
                if board_contain_shapes.__getitem__((i, j)) in set_shapes)

    def axis_directions_gen(self):
        for axis_dirs in self.full_directions:
            yield axis_dirs

    def directions_gen(self):
        for axis_dirs in self.full_directions:
            for direction in axis_dirs:
                yield direction


class AbstractMatchesSearcher(ABC):

    @abstractmethod
    def scan_board_for_matches(self, board: Board):
        pass


class MatchesSearcher(AbstractSearcher):

    def __init__(self, length, board_ndim):
        self.__3length, self.__4length, self.__5length = range(2, 5)
        self.stop_event = threading.Event()

        super().__init__(board_ndim)

    # def scan_board_for_matches(self, board: Board, need_all: bool = True, checking_point: list[Point] = []):
    #     matches = set()
    #     new_power_ups = dict()
    #     self.stop_event = threading.Event()

    #     lst = []
    #     s_t = time.time()
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=90) as executor:

    #             # if not need_all:
    #             #     assert checking_point is not None, "checking_point must have if need_all is False"
    #             #     if point not in checking_point:
    #             #         continue
    #             futures = [executor.submit(self.__get_match3_for_point, board, point, need_all=need_all) for point in self.points_generator(board) if (need_all or (not need_all and point in checking_point))]
    #             done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

    #             for future in concurrent.futures.as_completed(futures):
    #                 to_del, to_add = future.result()
    #                 if to_del:
    #                     lst.append((to_del, to_add))
    #                     if not need_all:
    #                         print("set stop event")
    #                         self.stop_event.set()
    #                         break

    #     e_t = time.time()
    #     print("scan board", e_t - s_t)
    #     for i in range(len(lst)):
    #         matches.update(lst[i][0])
    #         new_power_ups.update(lst[i][1])

    #     return matches, new_power_ups

    def scan_board_for_matches(self, board: Board, need_all: bool = True,
                               checking_point: list[Point] = None, focus_range=None):
        matches = set()
        new_power_ups = dict()

        board_rows, board_cols = board.board_size
        board_contain_shapes = board.board

        if not need_all:
            assert checking_point is not None, "checking_point must have if need_all is False"
            lst_points = checking_point
        else:
            lst_points = self.generate_movable_point(board_rows, board_cols, board_contain_shapes, focus_range)

        for point in lst_points:
            to_del, to_add = self.__get_match3_for_point(
                board_rows, board_cols, board_contain_shapes, point, need_all=need_all
            )

            if to_del:
                matches.update(to_del)
                new_power_ups.update(to_add)
                if not need_all:
                    break
        return matches, new_power_ups

    def __get_match3_for_point(self, board_rows, board_cols, board_contain_shapes, point: Point, need_all: bool = True):
        shape = board_contain_shapes.__getitem__(point.get_coord())
        match3_list = []
        power_up_list: dict[Point, int] = {}
        early_stop = False

        search_directions = self.normal_directions + self.plane_directions if (not need_all) else self.full_directions

        for neighbours, length, idx in cfunctions.generator_neighbours(board_rows, board_cols, board_contain_shapes,
                   *point.get_coord(), shape, search_directions, early_stop):
            match3_list.extend([Cell(n[0], n[1], n[2]) for n in neighbours])

            if not need_all:
                early_stop = True

            if length > 2 and idx != -1 and isinstance(point, Point):
                if point in power_up_list.keys():
                    power_up_list[point] = max(
                        power_up_list[point], self.get_power_up_type(idx)
                    )
                else:
                    power_up_list[point] = self.get_power_up_type(idx)

        if len(match3_list) > 0:
            match3_list.append(Cell(shape, *point.get_coord()))
        return match3_list, power_up_list

    def __generator_neighbours(
        self,
        board_rows, board_cols,
        board_contain_shapes,
        point: Point,
        filter_shape,
        early_stop: bool = False,
        only_2_matches: bool = False,
    ):
        curRow, curCol = point.get_coord()

        lst_cells = []

        for idx, axis_dirs in enumerate(self.normal_directions + self.plane_directions
                                        if only_2_matches else self.full_directions):
            newCells = []
            for dir_ in axis_dirs:
                newRow, newCol = curRow + dir_[0], curCol + dir_[1]
                if not is_valid_point(newRow, newCol, board_rows, board_cols):
                    lst_cells.append(([], 0, -1))
                    break

                # Get shape from board
                shape = board_contain_shapes.__getitem__((newRow, newCol))
                if shape != filter_shape:
                    continue

                cell = Cell(shape, newRow, newCol)
                newCells.append(cell)
            else:
                lst_cells.append((newCells, len(axis_dirs), idx))

            if early_stop:
                break

        return lst_cells

    @staticmethod
    def __filter_cells_by_shape(shape, cells):
        return [cell for cell in cells if cell.shape == shape]

class AbstractMonster(ABC):
    def __init__(
        self,
        relax_interval,
        setup_interval=0,
        position: Point = None,
        hp=30,
        width: int = 1,
        height: int = 1,
        have_paper_box: bool = False,
        request_masked: list[int] = None
    ):
        self.real_monster = True
        self._hp = hp
        self._origin_hp = hp
        self._progress = 0
        self._relax_interval = relax_interval
        self._setup_interval = setup_interval
        self._position = position
        self._width, self._height = width, height
        self.have_paper_box = have_paper_box
        self.escape_paper_box = True
        if self.have_paper_box:
            self._paper_box_hp = 0
            self.escape_paper_box = True

        self.__left_dmg_mask = self.__get_left_mask(self._position, self._height)
        self.__right_dmg_mask = self.__get_right_mask(
            self._position + Point(0, self._width - 1), self._height
        )
        self.__top_dmg_mask = self.__get_top_mask(self._position, self._width)
        self.__down_dmg_mask = self.__get_down_mask(
            self._position + Point(self._height - 1, 0), self._width
        )

        self.__inside_dmg_mask = [
            Point(i, j) + position
            for i, j in product(range(self._height), range(self._width))
        ]
        self.cause_dmg_mask = []
        if request_masked is not None and len(request_masked) == 5:
            self.available_mask = request_masked
        else:
            self.available_mask = [1, 1, 1, 1, 1]  # left, right, top, down, inside

    @property
    def dmg_mask(self):
        return (
            (self.__left_dmg_mask if self.available_mask[0] else [])
            + (self.__right_dmg_mask if self.available_mask[1] else [])
            + (self.__top_dmg_mask if self.available_mask[2] else [])
            + (self.__down_dmg_mask if self.available_mask[3] else [])
        )

    @property
    def inside_dmg_mask(self):
        return self.__inside_dmg_mask if self.available_mask[4] else []

    @property
    def mons_positions(self):
        return self.__inside_dmg_mask

    @abstractmethod
    def act(self):
        self._progress += 1

    def set_position(self, position: Point):
        self._position = position
        # Update new damage mask
        self.__left_dmg_mask = self.__get_left_mask(self._position, self._height)
        self.__right_dmg_mask = self.__get_right_mask(
            self._position + Point(0, self._width - 1), self._height
        )
        self.__top_dmg_mask = self.__get_top_mask(self._position, self._width)
        self.__down_dmg_mask = self.__get_down_mask(
            self._position + Point(self._height - 1, 0), self._width
        )

        self.__inside_dmg_mask = [
            Point(i, j) + position
            for i, j in product(range(self._height), range(self._width))
        ]

    def attacked(self, match_damage, pu_damage):
        if self.have_paper_box and self._paper_box_hp > 0:
            self._paper_box_hp -= 1 if match_damage > 0 else 0
            if self._paper_box_hp == 0:
                self.escape_paper_box = True
        else:
            damage = match_damage + pu_damage

            assert self._hp > 0, f"self._hp need to be positive, but self._hp = {self._hp}"
            self._hp -= damage

    @staticmethod
    def __get_left_mask(point: Point, height: int):
        mask = []
        for i in range(height):
            _point = point + Point(i, -1)
            if _point.get_coord()[0] >= 0 and _point.get_coord()[1] >= 0:
                mask.append(_point)
        return mask

    @staticmethod
    def __get_top_mask(point: Point, width: int):
        mask = []
        for i in range(width):
            _point = point + Point(-1, i)
            if _point.get_coord()[0] >= 0 and _point.get_coord()[1] >= 0:
                mask.append(_point)
        return mask

    @staticmethod
    def __get_right_mask(point: Point, height: int):
        mask = []
        for i in range(height):
            mask.append(point + Point(i, 1))
        return mask

    @staticmethod
    def __get_down_mask(point: Point, width: int):
        mask = []
        for i in range(width):
            mask.append(point + Point(1, i))
        return mask

    def get_hp(self):
        return self._hp

    def get_dame(self, matches, brokens, disco_brokens):
        """
        return: match_damage, pu_damage
        """
        __matches = [ele.point for ele in matches]
        __disco_brokens = [(ele.point if isinstance(ele, Cell) else ele) for ele in disco_brokens]
        mons_inside_dmg = 0
        for coor in brokens:
            if isinstance(coor, Cell):
                coor = coor.point
            if coor in set(self.inside_dmg_mask):
                mons_inside_dmg += 1

        # print(self.available_mask)
        # print(self.__right_dmg_mask)
        # print(self.__down_dmg_mask)

        return len(set(self.dmg_mask) & set(__matches)), \
            mons_inside_dmg + len(set(self.dmg_mask) & set(__disco_brokens))


class DameMonster(AbstractMonster):
    def __init__(
        self,
        position: Point,
        relax_interval=6,
        setup_interval=3,
        hp=20,
        width: int = 1,
        height: int = 1,
        dame=4,
        cancel_dame=5,
        have_paper_box: bool = False,
        request_masked: list[int] = None
    ):
        super().__init__(relax_interval, setup_interval, position, hp, width, height, have_paper_box, request_masked=request_masked)

        self._damage = dame

        self._cancel = cancel_dame
        self._cancel_dame = cancel_dame

    def act(self):
        super().act()
        if not self.have_paper_box:
            if self._cancel <= 0:
                self._progress = 0
                self._hp += self._cancel  # because of negative __cancel
                self._cancel = self._cancel_dame
                return {
                    "damage": 0,
                    "cancel_score": 2,
                }
        else:
            if self._paper_box_hp <= 0:
                self.available_mask = [1, 1, 1, 1, 1]
                if self.escape_paper_box:
                    self._progress = 0
                    self.escape_paper_box = False

        if self._progress > self._relax_interval + self._setup_interval:
            self._progress = 0
            return {"damage": self._damage}

        return {"damage": 0}

    def attacked(self, match_damage, pu_damage):
        damage = match_damage + pu_damage

        if (
            self.have_paper_box and self._progress >= self._relax_interval + self._setup_interval
        ):
            if self._paper_box_hp <= 0:
                self._paper_box_hp = 3
                self.available_mask = [1, 1, 1, 1, 0]

        super().attacked(match_damage, pu_damage)


class BoxMonster(AbstractMonster):
    def __init__(
        self,
        box_mons_type: int,
        position: Point,
        relax_interval: int = 8,
        setup_interval: int = 0,
        hp=30,
        width: int = 1,
        height: int = 1,
        have_paper_box: bool = False,
    ):
        super().__init__(relax_interval, 0, position, hp, width, height, have_paper_box)
        self.__box_monster_type = box_mons_type

    def act(self):
        super().act()
        if self._progress > self._relax_interval + self._setup_interval:
            self._progress = 0
            if self.__box_monster_type == GameObject.monster_box_box:
                return {"box": GameObject.blocker_box}
            if self.__box_monster_type == GameObject.monster_box_bomb:
                return {"box": GameObject.blocker_bomb}
            if self.__box_monster_type == GameObject.monster_box_thorny:
                return {"box": GameObject.blocker_thorny}
            if self.__box_monster_type == GameObject.monster_box_both:
                return {
                    "box": (
                        GameObject.blocker_bomb
                        if np.random.uniform(0, 1.0) <= 0.5
                        else GameObject.blocker_thorny
                    )
                }
        return {}


class BombBlocker(DameMonster):
    def __init__(
        self,
        position: Point,
        relax_interval=3,
        setup_interval=0,
        hp=2,
        width: int = 1,
        height: int = 1,
        dame=2,
        cancel_dame=5,
    ):
        super().__init__(
            position,
            relax_interval,
            setup_interval,
            hp,
            width,
            height,
            dame,
            cancel_dame,
        )

        self.is_box = True if dame == 0 else False
        self.real_monster = False

    def act(self):
        if self._progress > self._relax_interval + self._setup_interval:
            self._progress = 0
            self._hp = -999
            return {"damage": self._damage}

        return {"damage": 0}

    def attacked(self, match_damage, pu_damage):
        return super().attacked(match_damage, pu_damage)


class ThornyBlocker(DameMonster):
    def __init__(
        self,
        position: Point,
        relax_interval=999,
        setup_interval=999,
        hp=1,
        width: int = 1,
        height: int = 1,
        dame=1,
        cancel_dame=5,
    ):
        super().__init__(
            position,
            relax_interval,
            setup_interval,
            hp,
            width,
            height,
            dame,
            cancel_dame,
        )

        self.real_monster = False

    def act(self):
        if self._progress > self._relax_interval + self._setup_interval:
            self._progress = 0
            self._hp = -999
            return {"damage": self._damage}

        return {"damage": 0}

    def attacked(self, match_damage, pu_damage):
        if pu_damage > 0:
            self._hp = -999
        elif match_damage > 0:
            self._progress = self._relax_interval + self._setup_interval + 1


class AbstractPowerUpActivator(ABC):
    @abstractmethod
    def activate_power_up(self, power_up_type: int, point: Point, board: Board):
        pass


class PowerUpActivator(AbstractPowerUpActivator):
    def __init__(self):
        self.__bomb_affect = self.__get_bomb_affect()
        self.__plane_affect = self.__get_plane_affect()

    def activate_power_up(self, point: Point, directions, board: Board, list_monsters: list[AbstractMonster]):
        return_brokens, disco_brokens = set(), set()
        brokens = []
        point2 = point + directions
        shape1 = board.get_shape(point)
        shape2 = board.get_shape(point2)

        swap_type = None

        if shape1 in GameObject.set_powers_shape and shape2 in GameObject.set_powers_shape:

            #adding both points to return_brokens
            return_brokens.add(point)
            return_brokens.add(point2)

            # Merge power_up
            if shape1 <= shape2:
                shape1, shape2 = shape2, shape1
                point, point2 = point2, point

            # With disco
            if shape1 == GameObject.power_disco:
                if shape2 != GameObject.power_disco:
                    if shape2 == GameObject.power_missile_h or shape2 == GameObject.power_missile_v:
                        swap_type = GameAction.swap_merge_missile_disco
                    elif shape2 == GameObject.power_bomb:
                        swap_type = GameAction.swap_merge_bomb_disco
                    elif shape2 == GameObject.power_plane:
                        swap_type = GameAction.swap_merge_plane_disco                    
                    
                    chosen_color = np.random.randint(1, board.n_shapes + 1)
                    # print("Chosen Color For merging", chosen_color)
                    for i in range(board.board_size[0]):
                        for j in range(board.board_size[1]):
                            _p = Point(i, j)
                            if board.get_shape(_p) == chosen_color:
                                return_brokens.add(_p)
                                disco_brokens.add(_p)
                                shape2 = shape2 if (shape2 > GameObject.power_missile_v) else random.choice([GameObject.power_missile_h, GameObject.power_missile_v])
                                brokens.extend(self.__activate_not_merge(shape2, _p, board, list_monsters, None))
                else:
                    swap_type = GameAction.swap_merge_disco_disco
                    brokens = [Point(i, j) for i, j in product(range(board.board_size[0]), range(board.board_size[1]))]
                    disco_brokens = set(brokens)

            # With plane
            elif shape1 == GameObject.power_plane:
                for _dir in self.__plane_affect:
                    brokens.append(point + Point(*_dir))
                mons_pos = board.get_monster()
                try:
                    if shape2 != GameObject.power_plane:
                        if shape2 == GameObject.power_missile_h or shape2 == GameObject.power_missile_v:
                            swap_type = GameAction.swap_merge_missile_plane
                        elif shape2 == GameObject.power_bomb:
                            swap_type = GameAction.swap_merge_bomb_plane
                        random_mons = mons_pos[np.random.randint(0, len(mons_pos))]
                        brokens.append(random_mons)
                        brokens.append(random_mons) # Because of self activate of another PU
                        brokens.extend(self.__activate_not_merge(shape2, random_mons, board, list_monsters, None))
                    else:
                        swap_type = GameAction.swap_merge_plane_plane
                        brokens.extend(random.sample(mons_pos, 6) if len(mons_pos) > 6 else mons_pos)
                except:
                    # print("No Monster on Board")
                    print(board)

            # With bomb
            elif shape1 == GameObject.power_bomb:
                if shape2  != GameObject.power_bomb:
                    if shape2 == GameObject.power_missile_h or shape2 == GameObject.power_missile_v:
                        swap_type = GameAction.swap_merge_missile_bomb
                    for i in range(-1,2,1):
                        brokens.append(point + Point(i,0)) # Because of self activate of another PU
                        brokens.extend(self.__activate_not_merge(shape2, point + Point(i,0), board, list_monsters, None))
                        brokens.append(point + Point(0,i)) # Because of self activate of another PU
                        brokens.extend(self.__activate_not_merge(shape2, point + Point(0,i), board, list_monsters, None))
                else:
                    brokens.append(point)
                    swap_type = GameAction.swap_merge_bomb_bomb
                    def check_for_diagonal(i, _h, _v, point):
                        out_lst = []
                        prev_point = point
                        for _ in range(1, i):
                            cur_point = prev_point + Point(_h, 0)
                            if not self.check_shield(prev_point, cur_point, board, list_monsters):
                                out_lst.append(cur_point)
                                prev_point = cur_point
                            else:
                                break
                        
                        prev_point = point
                        for _ in range(1, i):
                            cur_point = prev_point + Point(0, _v)
                            if not self.check_shield(prev_point, cur_point, board, list_monsters):
                                out_lst.append(cur_point)
                                prev_point = cur_point
                            else:
                                break
                            
                        if not self.check_shield(point, point + Point(_h, _v), board, list_monsters):
                            cur_point = point + Point(_h, _v)
                            out_lst.append(cur_point)
                            out_lst.extend(check_for_diagonal(i - 1, _h, _v, cur_point))
                            
                        return out_lst
                    is_ne_v = is_ne_h = is_po_v = is_po_h = True
                    for i in range(1, 5):
                        prev_coeff = i - 1
                        
                        # check for vertical
                        if is_po_v and not self.check_shield(point + Point(prev_coeff, 0), point + Point(i, 0), board, list_monsters):
                            brokens.append(point + Point(i, 0))
                        else:
                            is_po_v = False
                        if is_ne_v and not self.check_shield(point + Point(-prev_coeff, 0), point + Point(-i, 0), board, list_monsters):
                            brokens.append(point + Point(-i, 0))
                        else:
                            is_ne_v = False
                        #check for horizontal
                        if is_po_h and not self.check_shield(point + Point(0, prev_coeff), point + Point(0, i), board, list_monsters):
                            brokens.append(point + Point(0, i))
                        else:
                            is_po_h = False
                        if is_ne_h and not self.check_shield(point + Point(0, -prev_coeff), point + Point(0, -i), board, list_monsters):
                            brokens.append(point + Point(0, -i))
                        else:
                            is_ne_h = False

                    # check for diagonal
                    for i in range(-1, 2, 2):
                        for j in range(-1, 2, 2):
                            prev_point = point
                            cur_point = point + Point(i, j)
                            if not self.check_shield(prev_point, cur_point, board, list_monsters):
                                brokens.append(cur_point)
                                prev_point = cur_point
                                for _ in range(1, 5):
                                    tmp_prev_point = prev_point
                                    tmp_cur_point = tmp_prev_point + Point(i, 0)
                                    if not self.check_shield(tmp_prev_point, tmp_cur_point, board, list_monsters):
                                        brokens.append(tmp_cur_point)
                                        tmp_prev_point = tmp_cur_point
                                    else:
                                        break
                                for _ in range(1, 5):
                                    tmp_prev_point = prev_point
                                    tmp_cur_point = tmp_prev_point + Point(0, j)
                                    if not self.check_shield(tmp_prev_point, tmp_cur_point, board, list_monsters):
                                        brokens.append(tmp_cur_point)
                                        tmp_prev_point = tmp_cur_point
                                    else:
                                        break
                                if not self.check_shield(prev_point, prev_point + Point(i, j), board, list_monsters):
                                    cur_point = prev_point + Point(i, j)
                                    brokens.append(cur_point)
                                    brokens.extend(check_for_diagonal(3, i, j, cur_point))
                            else:
                                continue
            # With missiles
            else:
                swap_type = GameAction.swap_merge_missile_missile
                brokens.append(point)
                brokens.extend(self.__activate_not_merge( GameObject.power_missile_h, point, board, list_monsters, None))
                brokens.extend(self.__activate_not_merge( GameObject.power_missile_v, point, board, list_monsters, None))

        elif shape1 in GameObject.set_powers_shape:
            return_brokens.add(point)
            if shape1 == GameObject.power_disco:
                swap_type = GameAction.swap_power_disco
                disco_brokens |= set(
                    self.__activate_not_merge(shape1, point, board, list_monsters, shape2)
                )
            else:
                if shape1 == GameObject.power_missile_h:
                    swap_type = GameAction.swap_power_missile_h
                elif shape1 == GameObject.power_missile_v:
                    swap_type = GameAction.swap_power_missile_v
                elif shape1 == GameObject.power_bomb:
                    swap_type = GameAction.swap_power_bomb
                elif shape1 == GameObject.power_plane:
                    swap_type = GameAction.swap_power_plane
                brokens = self.__activate_not_merge(shape1, point, board, list_monsters, shape2)

        elif shape2 in GameObject.set_powers_shape:
            return_brokens.add(point2)
            if shape2 == GameObject.power_disco:
                swap_type = GameAction.swap_power_disco
                disco_brokens |= set(
                    self.__activate_not_merge(shape2, point2, board, list_monsters, shape1)
                )
            else:
                if shape2 == GameObject.power_missile_h:
                    swap_type = GameAction.swap_power_missile_h
                elif shape2 == GameObject.power_missile_v:
                    swap_type = GameAction.swap_power_missile_v
                elif shape2 == GameObject.power_bomb:
                    swap_type = GameAction.swap_power_bomb
                elif shape2 == GameObject.power_plane:
                    swap_type = GameAction.swap_power_plane
                brokens = self.__activate_not_merge(shape2, point2, board, list_monsters, shape1)

        inside_brokens = copy.deepcopy(brokens)
        brokens = list(set(brokens))
        while brokens:
            try:
                consider_point = brokens.pop(0)
                # print(consider_point)
                if consider_point in return_brokens:
                    continue
                shape_c = board.get_shape(consider_point)
                if shape_c == board.immovable_shape:
                    continue
                return_brokens.add(consider_point)
                if shape_c in GameObject.set_powers_shape:
                    if shape_c == GameObject.power_disco:
                        disco_brokens |= set(
                            self.__activate_not_merge(
                                shape_c, consider_point, board, list_monsters, shape1
                            )
                        )
                    else:
                        more_pu =   self.__activate_not_merge(
                                shape_c, consider_point, board, list_monsters, shape1
                            )
                        brokens.extend(more_pu)
                        inside_brokens.extend(more_pu)
                        brokens = list(set(brokens))
            except OutOfBoardError:
                continue

        return return_brokens, disco_brokens, inside_brokens, swap_type

    # def __activate_merge(
    #     self,
    #     shape1: int,
    #     shape2: int,
    # ):
    #     pass

    def get_dir(self, prev_point: Point, cur_point: Point):
        direction: Point = cur_point - prev_point
        x, y = direction.get_coord()
        # left, right, top, down, inside
        if y == 1 and x == 1:
            return [0, 2]
        if y == 1 and x == -1:
            return [0, 3]
        if y == -1 and x == 1:
            return [1, 2]
        if y == -1 and x == -1:
            return [1, 3]
        if y == 1:
            return [0]
        if y == -1:
            return [1]
        if x == 1:
            return [2]
        if x == -1:
            return [3]

    def check_shield(self, prev_point: Point, cur_point: Point, board: Board, list_monsters: list[AbstractMonster]):
        try:
            if prev_point == cur_point:
                return False
            shape1 = board.get_shape(prev_point)
            shape2 = board.get_shape(cur_point)
            if shape1 not in GameObject.set_monsters_shape or shape2 in GameObject.set_monsters_shape:
                affect_dirs = self.get_dir(prev_point, cur_point)
                for mons in list_monsters:
                    if cur_point in mons.mons_positions:
                        for __adir in affect_dirs:
                            if mons.available_mask[__adir] == 0:
                                return True
            return False
        except OutOfBoardError:
            return True

    def __activate_not_merge(
        self, power_up_type: int, point: Point, board: Board, list_monsters, _color: int = None
    ):
        brokens = []
        # print("Power up to explode", power_up_type, point)
        if power_up_type == GameObject.power_plane:
            for _dir in self.__plane_affect:
                cur_point = point + Point(*_dir)
                if not self.check_shield(point, cur_point, board, list_monsters):
                    brokens.append(cur_point)

            mons_pos = board.get_monster()
            try:
                brokens.append(mons_pos[np.random.randint(0, len(mons_pos))])
            except:
                print("No Monster on Board")
                print(board)

        elif power_up_type == GameObject.power_missile_h:
            pos = point.get_coord()
            prev_point = point
            #backward move
            for i in range(pos[1] - 1, -1, -1):
                cur_point = Point(pos[0], i)
                if not self.check_shield(prev_point, cur_point, board, list_monsters):
                    brokens.append(cur_point)
                else:
                    break
                prev_point = cur_point
            #forward move
            prev_point = point
            for i in range(pos[1] + 1, board.board_size[1]):
                cur_point = Point(pos[0], i)
                if not self.check_shield(prev_point, cur_point, board, list_monsters):
                    brokens.append(cur_point)
                else:
                    break
                prev_point = cur_point
        elif power_up_type == GameObject.power_missile_v:
            pos = point.get_coord()
            prev_point = point
            #backward move
            for i in range(pos[0] - 1, -1, -1):
                cur_point = Point(i, pos[1])
                if not self.check_shield(prev_point, cur_point, board, list_monsters):
                    brokens.append(cur_point)
                else:
                    break
                prev_point = cur_point
            #forward move
            prev_point = point
            for i in range(pos[0] + 1, board.board_size[0]):
                cur_point = Point(i, pos[1])
                if not self.check_shield(prev_point, cur_point, board, list_monsters):
                    brokens.append(cur_point)
                else:
                    break
                prev_point = cur_point
        elif power_up_type == GameObject.power_bomb:
            is_ne_v = is_ne_h = is_po_v = is_po_h = True
            for i in range(1, 3):
                prev_coeff = i - 1

                # check for vertical
                if is_po_v and not self.check_shield(point + Point(prev_coeff, 0), point + Point(i, 0), board, list_monsters):
                    brokens.append(point + Point(i, 0))
                else:
                    is_po_v = False
                if is_ne_v and not self.check_shield(point + Point(-prev_coeff, 0), point + Point(-i, 0), board, list_monsters):
                    brokens.append(point + Point(-i, 0))
                else:
                    is_ne_v = False
                #check for horizontal
                if is_po_h and not self.check_shield(point + Point(0, prev_coeff), point + Point(0, i), board, list_monsters):
                    brokens.append(point + Point(0, i))
                else:
                    is_po_h = False
                if is_ne_h and not self.check_shield(point + Point(0, -prev_coeff), point + Point(0, -i), board, list_monsters):
                    brokens.append(point + Point(0, -i))
                else:
                    is_ne_h = False

            # check for diagonal
            for i in range(-1, 2, 2):
                for j in range(-1, 2, 2):
                    prev_point = point
                    cur_point = point + Point(i, j)
                    if not self.check_shield(prev_point, cur_point, board, list_monsters):
                        brokens.append(cur_point)
                        prev_point = cur_point

                        if not self.check_shield(prev_point, prev_point + Point(i, 0), board, list_monsters):
                            brokens.append(prev_point + Point(i, 0))
                        if not self.check_shield(prev_point, prev_point + Point(0, j), board, list_monsters):
                            brokens.append(prev_point + Point(0, j))
                        if not self.check_shield(prev_point, prev_point + Point(i, j), board, list_monsters):
                            brokens.append(prev_point + Point(i, j))
                    else:
                        continue


        elif power_up_type == GameObject.power_disco:
            assert _color is not None, "Disco Power Up need color to be cleared"
            for i in range(board.board_size[0]):
                for j in range(board.board_size[1]):
                    _p = Point(i, j)
                    if board.get_shape(_p) == _color:
                        brokens.append(Cell(_color, *_p.get_coord()))
        else:
            raise ValueError(f"Do not have any power up type {power_up_type}")
        return brokens

    def __activate_merge(self, point1: Point, point2: Point, board: Board):
        pass

    @staticmethod
    def __get_plane_affect():
        affects = [[0 for _ in range(2)] for _ in range(4)]
        affects[0][0] = 1
        affects[1][0] = -1
        affects[2][1] = 1
        affects[3][1] = -1

        return affects

    @staticmethod
    def __get_bomb_affect():
        affects = [[i - 3, j - 3] for i, j in product(range(5), range(5))]

        return affects


class AbstractMovesSearcher(ABC):

    @abstractmethod
    def search_moves(self, board: Board):
        pass


class MovesSearcher(AbstractMovesSearcher, MatchesSearcher):
    def search_moves(self, board: Board, all_moves=False):
        possible_moves = set()
        not_have_pu = True

        board_rows, board_cols = board.board_size
        board_contain_shapes = board.board

        # check for powerup activation
        for cur_row, cur_col in board.power_points:
            # This point was valid in points generator -> get shape instantly
            if board_contain_shapes.__getitem__((cur_row, cur_col)) in GameObject.set_powers_shape:
                for direction in self.directions_gen():
                    new_row, new_col = cur_row + direction[0], cur_col + direction[1]
                    if not is_valid_point(new_row, new_col, board_rows, board_cols):
                        continue

                    new_shape = board_contain_shapes.__getitem__((new_row, new_col))

                    if new_shape in GameObject.set_unmovable_shape:
                        continue

                    # board.move(point, Point(*direction))
                    # # inverse move
                    # board.move(point, Point(*direction))
                    elif new_shape in GameObject.set_movable_shape:
                        not_have_pu = False
                        possible_moves.add((Point(cur_row, cur_col), tuple(direction)))
                        if not all_moves:
                            break

                if not all_moves and not not_have_pu:
                    break

        if all_moves is True or (all_moves is False and not_have_pu):
            for point in self.generate_movable_point(board_rows, board_cols, board_contain_shapes,
                                                     None, GameObject.set_tiles_shape):
                possible_moves_for_point = self.__search_moves_for_point(
                    board, point, need_all=all_moves
                )

                possible_moves.update(possible_moves_for_point)
                if len(possible_moves_for_point) > 0 and not all_moves:
                    break

        return possible_moves

    def __search_moves_for_point(self, board: Board, point: Point, need_all=True):
        # contain tuples of point and direction

        possible_moves = set()

        board_rows, board_cols = board.board_size
        board_contain_shapes = board.board

        cur_row, cur_col = point.get_coord()
        for direction in self.directions_gen():
            new_row, new_col = cur_row + direction[0], cur_col + direction[1]

            if not is_valid_point(new_row, new_col, board_rows, board_cols):
                continue

            new_shape = board_contain_shapes.__getitem__((new_row, new_col))
            if new_shape in GameObject.set_unmovable_shape:
                continue

            board_contain_shapes[(cur_row, cur_col)], board_contain_shapes[(new_row, new_col)] \
                = board_contain_shapes[(new_row, new_col)], board_contain_shapes[(cur_row, cur_col)]

            if need_all is False:
                checking_point = [point, Point(new_row, new_col)]
            matches, _ = self.scan_board_for_matches(board, need_all=need_all, checking_point=checking_point)

            # inverse move
            board_contain_shapes[(cur_row, cur_col)], board_contain_shapes[(new_row, new_col)] \
                = board_contain_shapes[(new_row, new_col)], board_contain_shapes[(cur_row, cur_col)]

            if len(matches) > 0:
                possible_moves.add((point, tuple(direction)))
                if not need_all:
                    break

        return possible_moves


class AbstractFiller(ABC):

    @abstractmethod
    def move_and_fill(self, board):
        pass


class Filler(AbstractFiller):

    def __init__(self, random_state=None):
        self.__random_state = random_state
        np.random.seed(self.__random_state)

    def move_and_fill(self, board: Board):
        self.__move_nans(board)
        self.__fill(board)

    def __move_nans(self, board: Board):
        _, cols = board.board_size
        for col_ind in range(cols):
            line = board.get_line(col_ind)
            if np.any(np.isnan(line)):
                new_line = self._move_line(line, board.immovable_shape)
                board.put_line(col_ind, new_line)
            else:
                continue

    @staticmethod
    def _move_line(line, immovable_shape):
        num_of_nans = np.isnan(line).sum()
        immov_mask = mask_immov_mask(line, immovable_shape)
        nans_mask = np.isnan(line)
        new_line = np.zeros_like(line)
        new_line = np.where(immov_mask, line, new_line)

        num_putted = 0
        for ind, shape in enumerate(new_line):
            if (
                shape != immovable_shape
                and shape
                not in GameObject.set_unmovable_shape
                and num_putted < num_of_nans
            ):
                new_line[ind] = np.nan
                num_putted += 1
                if num_putted == num_of_nans:
                    break

        spec_mask = nans_mask | immov_mask
        regular_values = line[~spec_mask]
        new_line[(new_line == 0)] = regular_values
        return new_line

    def __fill(self, board):
        is_nan_mask = np.isnan(board.board)
        num_of_nans = is_nan_mask.sum()

        new_shapes = np.random.randint(
            low=GameObject.color1, high=board.n_shapes + 1, size=num_of_nans
        )
        board.put_mask(is_nan_mask, new_shapes)


class AbstractPowerUpFactory(ABC):
    @abstractmethod
    def get_power_up_type(matches):
        pass


class PowerUpFactory(AbstractPowerUpFactory, AbstractSearcher):
    def __init__(self, board_ndim):
        super().__init__(board_ndim)


class BlockerFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_blocker(
        blocker_type: int, position: Point, width: int = 1, height: int = 1
    ):
        if blocker_type == GameObject.blocker_box:
            return BombBlocker(
                position, relax_interval=999, dame=0, width=width, height=height
            )
        elif blocker_type == GameObject.blocker_bomb:
            return BombBlocker(position, width=width, height=height)
        elif blocker_type == GameObject.blocker_thorny:
            return ThornyBlocker(position, width=width, height=height)


class AbstractGame(ABC):

    @abstractmethod
    def start(self, board):
        pass

    @abstractmethod
    def swap(self, point, point2):
        pass


class Game(AbstractGame):
    def __init__(
        self,
        rows,
        columns,
        n_shapes,
        length,
        player_hp=40,
        all_moves=False,
        immovable_shape=-1,
        random_state=None,
    ):
        self.board = Board(rows=rows, columns=columns, n_shapes=n_shapes, random_state=random_state)
        self.__max_player_hp = player_hp
        self.__player_hp = player_hp
        self.__random_state = random_state
        self.__immovable_shape = immovable_shape
        self.__all_moves = all_moves
        self.__mtch_searcher = MatchesSearcher(length=length, board_ndim=2)
        self.__mv_searcher = MovesSearcher(length=length, board_ndim=2)
        self.__filler = Filler(random_state=random_state)
        self.hit_rate, self.hit_dame = 0, 0
        self.__pu_activator = PowerUpActivator()

    def play(self, board: Union[np.ndarray, None]):
        self.start(board)
        while True:
            try:
                input_str = input()
                coords = input_str.split(", ")
                a, b, a1, b1 = [int(i) for i in coords]
                self.swap(Point(a, b), Point(a1, b1))
            except KeyboardInterrupt:
                break

    def start(
        self,
        board: Union[np.ndarray, None, Board],
        list_monsters: list[AbstractMonster],
    ):
        # TODO: check consistency of movable figures and n_shapes
        if board is None:
            rows, cols = self.board.board_size
            board = RandomBoard(rows, cols, self.board.n_shapes)
            board.set_random_board(random_state=self.__random_state)
            board = board.board
            self.board.set_board(board)
        elif isinstance(board, np.ndarray):
            self.board.set_board(board)
        elif isinstance(board, Board):
            self.board = board
        self.__operate_until_possible_moves()
        self.list_monsters = copy.deepcopy(list_monsters)
        self.num_mons = len(self.list_monsters)
        self.__player_hp = self.__max_player_hp

        return self

    def __start_random(self):
        rows, cols = self.board.board_size
        tmp_board = RandomBoard(rows, cols, self.board.n_shapes)
        tmp_board.set_random_board(random_state=self.__random_state)
        super().start(tmp_board.board)

    def swap(self, point: Point, point2: Point):
        direction = point2 - point
        try:
            score = self.__move(point, direction)

            return score
        except Exception as e:
            print(traceback.format_exc())
            print("Error when swaping", e)
            print("There was an error when swaping", [mon.get_hp() for mon in self.list_monsters])
            return {
                "board": self.board.board,
                "score": 0,
                "cancel_score": 0,
                "create_pu_score": 0,
                "match_damage_on_monster": 0,
                "power_damage_on_monster": 0,
                "damage_on_user": 0,
            }

    def __move(self, point: Point, direction: Point):
        score = 0
        rate_mons_hp = 0
        near_monster = 100
        cancel_score = 0
        create_pu_score = 0
        rate_match_dmg = 0
        rate_power_dmg = 0
        total_match_dmg = 0
        total_power_dmg = 0
        dmg = 0
        self_dmg = 0

        matches, new_power_ups, brokens, disco_brokens , inside_brokens, swap_type = self.__check_matches(point, direction)

        # Calculate scores
        score += len(brokens) + len(disco_brokens)

        for i in range(len(self.list_monsters)):
            near_monster = min(near_monster, point.euclidean_distance(self.list_monsters[i]._position))
            match_damage, pu_damage = self.list_monsters[i].get_dame(matches, inside_brokens, disco_brokens)
            rate_match_dmg += match_damage / self.list_monsters[i]._origin_hp
            rate_power_dmg += pu_damage / self.list_monsters[i]._origin_hp
            total_match_dmg += match_damage
            total_power_dmg += pu_damage
            score -= pu_damage

            self.list_monsters[i].attacked(match_damage, pu_damage)
            monster_result = self.list_monsters[i].act()

            rate_mons_hp += self.list_monsters[i].get_hp() / self.list_monsters[i]._origin_hp

        self.__player_hp -= self_dmg
        if len(matches) > 0 or len(brokens) > 0 or len(disco_brokens) > 0:
            score += len(matches)
            self.board.move(point, direction)
            if len(matches) > 0:
                self.board.delete(matches)
            if len(brokens) > 0:
                self.board.delete(brokens)
            if len(disco_brokens) > 0:
                self.board.delete(disco_brokens)

            ### Handle add power up
            for _point, _shape in new_power_ups.items():
                self.board.put_shape(_point, _shape)
                if _shape == GameObject.power_missile_h:
                    create_pu_score += 0.9
                if _shape == GameObject.power_missile_v:
                    create_pu_score += 1
                elif _shape == GameObject.power_plane:
                    create_pu_score += 1.5
                elif _shape == GameObject.power_bomb:
                    create_pu_score += 2.5
                elif _shape == GameObject.power_disco:
                    create_pu_score += 4.5
            
            ### Handle add power up and attack user
            if "box" in monster_result.keys():
                coor_x, coor_y = np.random.randint(0, [*self.board.board_size])
                while self.board.get_shape(Point(coor_x, coor_y)) in GameObject.set_unmovable_shape:
                    coor_x, coor_y = np.random.randint(0, [*self.board.board_size])

                mons_pos = Point(coor_x, coor_y)
                self.board.put_shape(mons_pos, monster_result["box"])
                self.list_monsters.append(
                    BlockerFactory.create_blocker(monster_result["box"], mons_pos)
                )
            if "damage" in monster_result.keys():
                self_dmg += monster_result["damage"]
                cancel_score += monster_result.get("cancel_score", 0)

            self.__filler.move_and_fill(self.board)
            self.__operate_until_possible_moves()
        reward = {
            "swap_type": swap_type,
            "score": score,
            "cancel_score": cancel_score,
            "near_monster": near_monster,
            "create_pu_score": create_pu_score,
            "rate_match_damage_on_monster": rate_match_dmg,
            "rate_power_damage_on_monster": rate_power_dmg,
            "match_damage_on_monster": total_match_dmg,
            "power_damage_on_monster": total_power_dmg,
            "rate_mons_hp": rate_mons_hp,
            "damage_on_user": self_dmg,
        }
        return reward

    def __check_matches(self, point: Point, direction: Point):
        tmp_board = self.__get_copy_of_board()
        tmp_board.move(point, direction)
        return_brokens, disco_brokens, inside_brokens, swap_type = self.__pu_activator.activate_power_up(
            point, direction, tmp_board, self.list_monsters
        )

        # if not disco_brokens.issubset(return_brokens):
        #     print(f"return_brokens: {return_brokens}, disco_brokens: {disco_brokens}, inside_brokens: {inside_brokens}")

        delete_points = return_brokens | disco_brokens
        if delete_points:
            tmp_board.delete(delete_points)
            self.__filler.move_and_fill(tmp_board)

        focus_range = self.__find_focus_range({
            point.get_coord(),
            (point + direction).get_coord(),
            *[p.get_coord() for p in delete_points]
        })

        matches, new_power_ups = self.__mtch_searcher.scan_board_for_matches(tmp_board, focus_range=focus_range)

        # # Test block to check focus range is valid
        # matches, new_power_ups = self.__mtch_searcher.scan_board_for_matches(tmp_board)
        # test_matches, test_new_power_ups = self.__mtch_searcher.scan_board_for_matches(tmp_board,
        #                                                                                focus_range=focus_range)
        # if matches != test_matches or new_power_ups != test_new_power_ups:
        #     print(f"__find_focus_range: ", focus_range)
        #     print(f"old_matches: {matches}")
        #     print(f"focus_range: {test_matches}")
        #     print("------------------------------------------------------------------------")

        return matches, new_power_ups, return_brokens, disco_brokens, inside_brokens, swap_type

    def _sweep_died_monster(self):
        mons_points = set()
        real_mons_alive, alive_flag, died_flag = False, False, False
        i = 0

        # print(self.board)
        # print("HP", [x.get_hp() for x in self.list_monsters])
        # print("real", [x.real_monster for x in self.list_monsters])

        while i < len(self.list_monsters):
            if self.list_monsters[i].get_hp() > 0:
                if self.list_monsters[i].real_monster:
                    real_mons_alive = True
                alive_flag = True
                i += 1
            else:
                died_flag = True
                mons_points.update(self.list_monsters[i].inside_dmg_mask)
                del self.list_monsters[i]

        # print("3 bools", alive_flag, real_mons_alive, died_flag)
        if alive_flag and real_mons_alive:
            if died_flag:
                self.board.delete(set(mons_points), allow_delete_monsters=True)

                self.__filler.move_and_fill(self.board)
                self.__operate_until_possible_moves()
        else:
            return True
        return False

    def __get_copy_of_board(self):
        return copy.deepcopy(self.board)

    def __operate_until_possible_moves(self):
        """
        scan board, then delete matches, move nans, fill
        repeat until no matches and appear possible moves
        """
        self.__scan_del_mvnans_fill_until()
        self.__shuffle_until_possible()
        return 0

    def __get_matches(self, focus_range=None):
        return self.__mtch_searcher.scan_board_for_matches(self.board, focus_range=focus_range)

    def __activate_power_up(self, power_up_type: int, point: Point):
        return self.__pu_activator.activate_power_up(power_up_type, point, self.board)

    def __get_possible_moves(self):
        return self.__mv_searcher.search_moves(self.board, all_moves=self.__all_moves)

    def __scan_del_mvnans_fill_until(self):
        score = 0
        matches, new_power_ups = self.__get_matches()

        old_matches = matches

        score += len(matches)
        while len(matches) > 0:
            self.board.delete(matches)
            for _point, _shape in new_power_ups.items():
                self.board.put_shape(_point, _shape)
            self.__filler.move_and_fill(self.board)

            focus_range = self.__find_focus_range([p.get_coord() for p in old_matches])
            matches, new_power_ups = self.__get_matches(focus_range)

            # Test block to check focus range is valid
            # matches, new_power_ups = self.__get_matches()
            # test_matches, test_new_power_ups = self.__get_matches(focus_range)
            #
            # if matches != test_matches or new_power_ups != test_new_power_ups:
            #     print(f"__find_focus_range: ", focus_range)
            #     print(f"old_matches: {old_matches} -> {matches}")
            #     print(f"old_matches: {old_matches} -> {test_matches}")
            #     print("------------------------------------------------------------------------")

            old_matches = matches

            score += len(matches)

        self.board.determine_power_points()

        return score

    @staticmethod
    def __find_focus_range(matches):
        max_row = -1
        start_col = 100
        end_col = -1

        for p in matches:
            max_row = max(max_row, p[0])
            start_col = min(start_col, p[1])
            end_col = max(end_col, p[1])

        return max_row + 2, start_col - 2, end_col + 2

    def __shuffle_until_possible(self):
        possible_moves = self.__get_possible_moves()
        while len(possible_moves) == 0:
            print("not have move")
            self.board.shuffle()
            self.__scan_del_mvnans_fill_until()
            possible_moves = self.__get_possible_moves()
        return self

    def get_player_hp(self):
        return self.__player_hp


class RandomGame(Game):

    def start(self, random_state=None, *args, **kwargs):
        rows, cols = self.board.board_size
        tmp_board = RandomBoard(rows, cols, self.board.n_shapes, random_state)
        tmp_board.set_random_board(random_state=random_state)
        super().start(tmp_board.board)
