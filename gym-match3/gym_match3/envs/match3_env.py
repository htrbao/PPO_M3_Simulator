import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

from gym_match3.envs.game import Game, Point
from gym_match3.envs.game import OutOfBoardError, ImmovableShapeError
from gym_match3.envs.levels import LEVELS, Match3Levels
from gym_match3.envs.renderer import Renderer
from gym_match3.envs.constants import GameObject
from gym_match3.envs.match3_helper import M3Helper

from itertools import product
import warnings
import time
import wandb
import numpy as np

BOARD_NDIM = 2


class Match3Env(gym.Env):
    metadata = {"render.modes": None}
    result_step = 0

    def __init__(
        self,
        rollout_len=100,
        all_moves=False,
        levels=None,
        random_state=None,
        obs_order: list[str] = [],
        level_group: tuple[int, int] = (0, 10),
    ):
        self.num_envs = 1
        self.rollout_len = rollout_len
        self.random_state = random_state
        self.all_moves = all_moves
        self.levels = levels or Match3Levels(LEVELS[level_group[0] : level_group[1]])
        print(
            f"This env manages level from group {level_group[0]} to group {level_group[1]}"
        )
        self.helper = M3Helper(10, 9, obs_order)
        self.h = self.levels.h
        self.w = self.levels.w
        self.n_shapes = self.levels.n_shapes
        self.__episode_counter = 0

        self.__game = Game(
            rows=self.h,
            columns=self.w,
            n_shapes=self.n_shapes,
            length=3,
            all_moves=all_moves,
            random_state=self.random_state,
        )
        self.reset()
        self.renderer = Renderer(self.n_shapes)

        # setting observation space
        self.observation_space = spaces.Box(
            low=GameObject.color1,
            high=self.n_shapes + 1,
            shape=(len(self.helper.obs_order), *self.__game.board.board_size),
            dtype=int,
        )

        # setting actions space
        self.__match3_actions = self.__get_available_actions_in_order()
        self.action_space = spaces.Discrete(len(self.__match3_actions))

    @staticmethod
    def __get_directions(board_ndim):
        """get available directions for any number of dimensions"""
        directions = [
            [[0 for _ in range(board_ndim)] for _ in range(2)]
            for _ in range(board_ndim)
        ]
        for ind in range(board_ndim):
            directions[ind][0][ind] = 1
            directions[ind][1][ind] = -1
        return directions

    def __points_generator(self):
        """iterates over points on the board"""
        rows, cols = self.__game.board.board_size
        points = [Point(i, j) for i, j in product(range(rows), range(cols))]
        for point in points:
            yield point

    def __get_available_actions(self):
        """calculate available actions for current board sizes"""
        actions = set()
        directions = self.__get_directions(board_ndim=BOARD_NDIM)
        for point in self.__points_generator():
            for axis_dirs in directions:
                for dir_ in axis_dirs:
                    dir_p = Point(*dir_)
                    new_point = point + dir_p
                    try:
                        _ = self.__game.board[new_point]
                        actions.add(frozenset((point, new_point)))
                    except OutOfBoardError:
                        continue
        return list(actions)

    def __get_available_actions_in_order(self):
        """calculate available actions for current board sizes in presented order"""
        actions = []
        rows, cols = self.__game.board.board_size
        for i in range(rows):
            for j in range(cols):
                dir_p = Point(0, 1)
                point = Point(i, j)
                new_point = point + dir_p
                try:
                    _ = self.__game.board[new_point]
                    actions.append(frozenset((point, new_point)))
                except OutOfBoardError:
                    continue
        for i in range(rows):
            for j in range(cols):
                dir_p = Point(1, 0)
                point = Point(i, j)
                new_point = point + dir_p
                try:
                    _ = self.__game.board[new_point]
                    actions.append(frozenset((point, new_point)))
                except OutOfBoardError:
                    continue
        return list(actions)

    def __get_action(self, ind):
        return self.__match3_actions[ind]

    def step(self, action):
        # make action
        m3_action = self.__get_action(action)
        # print(m3_action) #openlater
        ob = {}
        reward = self.__swap(*m3_action)
        is_early_done_game = self.__game._sweep_died_monster()

        # change counter even action wasn't successful
        self.__episode_counter += 1
        if (
            self.__episode_counter >= self.rollout_len
            or is_early_done_game
            or self.__game.get_player_hp() <= 0
        ):
            episode_over = True
            self.__episode_counter = 0

            if self.__game.get_player_hp() <= 0:
                reward.update(
                    {
                        "game": -2
                        - 1
                        * sum(
                            [
                                mon.get_hp() / mon._origin_hp
                                for mon in self.__game.list_monsters
                                if mon.real_monster
                            ]
                        )
                    }
                )
            else:
                reward.update(
                    {
                        "game": (
                            -1.5
                            - 1
                            * sum(
                                [
                                    mon.get_hp() / mon._origin_hp
                                    for mon in self.__game.list_monsters
                                    if mon.real_monster
                                ]
                            )
                            if not is_early_done_game
                            else 1.5 + 1 * self.__game.num_mons
                        )
                    }
                )

            # print(reward) #openlater
            self.result_step += 1
            obs, infos = self.reset()
            reward

            return obs, reward, episode_over, infos
        else:
            episode_over = False
            ob["board"] = self.__get_board()
            ob["list_monster"] = self.__game.list_monsters

        obs = self.helper._format_observation(ob["board"], ob["list_monster"], "cpu")
        # Check if non legal_action
        if 1 not in np.unique(obs["action_space"]):
            episode_over = True
            self.__episode_counter = 0
            reward.update({"game": -2.5})

            obs, infos = self.reset()
            return obs, reward, episode_over, infos

        return (
            self.helper.obs_to_tensor(obs["obs"]),
            reward,
            episode_over,
            {"action_space": obs["action_space"]},
        )

    def reset(self, *args, **kwargs):
        board, list_monsters = self.levels.next()
        self.__game.start(board, list_monsters)
        obs = self.helper._format_observation(self.__get_board(), list_monsters, "cpu")
        return self.helper.obs_to_tensor(obs["obs"]), {
            "action_space": obs["action_space"]
        }

    def __swap(self, point1, point2):
        try:
            reward = self.__game.swap(point1, point2)
        except ImmovableShapeError:
            reward = 0
        return reward

    def __get_board(self):
        return self.__game.board.board.copy()

    def render(self, action, mode="human", close=False):
        if close:
            warnings.warn("close=True isn't supported yet")
        # print(self.__game.board)
        tiles = self.__get_action(action)
        p1, p2 = tiles
        x1, y1 = p1.get_coord()
        x2, y2 = p2.get_coord()
        if x1 > x2 or y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        self.renderer.render_board(self.__game.board, {"x1": x1, "y1": y1, "x2": x2, "y2": y2})
