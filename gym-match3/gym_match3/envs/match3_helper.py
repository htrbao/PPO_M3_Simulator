import itertools

import numpy as np
import torch

from gym_match3.envs.constants import GameObject
from gym_match3.envs.game import AbstractMonster, ThornyBlocker


class M3Helper:
    def __init__(self, num_row: int = 10, num_col: int = 9, obs_order: list[str] = []) -> None:
        self.num_row = num_row
        self.num_col = num_col
        self.num_action = (self.num_row - 1) * self.num_col + self.num_row * (
            self.num_col - 1
        )
        if not obs_order:
            self.obs_order = [
                "none_tile",
                "color_1",
                "color_2",
                "color_3",
                "color_4",
                "color_5",
                "pu",
                "disco",
                "bomb",
                "missile_h",
                "missile_v",
                "plane",
                "blocker",
                "monster",
                "monster_match_dmg_mask",
                "monster_inside_dmg_mask",
                "self_dmg_mask",
                "match_normal",
                "match_2x2",
                "match_4_v",
                "match_4_h",
                "match_L",
                "match_T",
                "match_5",
                "legal_action",
                "heat_mask",
            ]
        else:
            self.obs_order = obs_order

    def _from_action_to_tile(self):
        a2t = {}
        max_h_action = (self.num_col - 1) * self.num_row
        for i in range(self.num_action):
            if i - max_h_action < 0:
                y, x = i % (self.num_col - 1), i // (self.num_col - 1)
                a2t[i] = {"x1": x, "y1": y, "x2": x, "y2": y + 1}
            else:

                y, x = (i - max_h_action) % self.num_col, (
                    i - max_h_action
                ) // self.num_col
                a2t[i] = {"x1": x, "y1": y, "x2": x + 1, "y2": y}

        return a2t

    def check_legal_pos_to_move(self, i: int, j: int, raw_board: np.array):
        return 0 <= i < self.num_row and 0 <= j < self.num_col and raw_board[i][j] in GameObject.set_movable_shape

    def check_required_tile(
            self,
            color,
            raw_board: np.array,
            i: int,
            j: int,
            check_type: list[tuple[int, int]],
    ):
        # if color_board[0][0] == 1:
        # print("\t",i, j)
        for x, y in check_type:
            # if color_board[0][0] == 1:
            # print("\t\t", x, y)
            check_row, check_col = i + x, j + y
            if (
                    not (0 <= check_row < self.num_row and 0 <= check_col < self.num_col)
                    or raw_board[i + x][j + y] != color
            ):

                return False

        return True

    def check_match(
            self,
            raw_board: np.array,
            match_normal: list[list[int]],
            match_2x2: list[list[int]],
            match_4_v: list[list[int]],
            match_4_h: list[list[int]],
            match_L: list[list[int]],
            match_T: list[list[int]],
            match_5: list[list[int]],
            legal_action: list[list[int]],
            action_space: list[int],
    ):
        check_types = (
            ((1, 0), (2, 0)),  # normal_XOO_v
            ((-1, 0), (1, 0)),  # normal_OXO_v
            ((-2, 0), (-1, 0)),  # normal_OOX_v
            ((0, 1), (0, 2)),  # normal_XOO
            ((0, -1), (0, 1)),  # 4. normal_OXO
            ((0, -2), (0, -1)),  # 5. normal_OOX
            ((0, -1), (-1, -1), (-1, 0)),  # 6. 2x2_wo_bottom_right
            ((0, -1), (1, -1), (1, 0)),  # 7. 2x2_wo_top_right
            ((0, 1), (-1, 1), (-1, 0)),  # 8. 2x2_wo_bottom_left
            ((0, 1), (1, 1), (1, 0)),  # 9. 2x2_wo_top_left
            ((0, -1), (0, 1), (0, 2)),  # 10. OXOO
            ((0, -2), (0, -1), (0, 1)),  # 11. OOXO
            ((-1, 0), (1, 0), (2, 0)),  # 12. OXOO_v
            ((-2, 0), (-1, 0), (1, 0)),  # 13. OOXO_v
            ((0, -2), (0, -1), (0, 1), (0, 2)),  # 14. OOXOO
            ((-2, 0), (-1, 0), (1, 0), (2, 0)),  # 15. OOXOO_v
            # match_L
            ((0, -1), (0, -2), (-2, 0), (-1, 0)),  # 16. 1st quarter
            ((0, -1), (0, -2), (1, 0), (2, 0)),  # 17. 2nd quarter
            ((0, 1), (0, 2), (1, 0), (2, 0)),  # 18. 3rd quarter
            ((0, 1), (0, 2), (-2, 0), (-1, 0)),  # 19. 4th quarter
            # match_T
            ((0, -1), (0, 1), (-1, 0), (-2, 0)),  # 20. up
            ((-1, 0), (1, 0), (0, 1), (0, 2)),  # 21. right
            ((0, 1), (0, -1), (1, 0), (2, 0)),  # 22. down
            ((1, 0), (-1, 0), (0, -1), (0, -2)),  # 23. left
        )

        oo = 1e9
        for i, j in itertools.product(range(self.num_row), range(self.num_col)):
            color = raw_board[i][j]
            if color not in GameObject.set_tiles_shape: continue

            raw_board[i][j] = -oo
            # wipe right
            if self.check_legal_pos_to_move(i, j + 1, raw_board):
                has_match_3 = False
                for type_c in [0, 1, 2, 3]:
                    if self.check_required_tile(
                            color, raw_board, i, j + 1, check_types[type_c]
                    ):
                        has_match_3 = True
                        legal_action[i][j] = 1
                        legal_action[i][j + 1] = 1
                        action_space[(self.num_col - 1) * i + j] = 1
                        # print((self.num_col - 1) * i + j)

                        match_normal[i][j + 1] = 1
                        for x, y in check_types[type_c]:
                            match_normal[i + x][j + 1 + y] = 1
                for type_c in [8, 9]:
                    if self.check_required_tile(
                            color, raw_board, i, j + 1, check_types[type_c]
                    ):
                        legal_action[i][j] = 1
                        legal_action[i][j + 1] = 1
                        action_space[(self.num_col - 1) * i + j] = 1
                        # print((self.num_col - 1) * i + j)

                        match_2x2[i][j + 1] = 1
                        for x, y in check_types[type_c]:
                            match_2x2[i + x][j + 1 + y] = 1
                        break
                if has_match_3:
                    for type_c in [12, 13]:
                        if self.check_required_tile(
                                color, raw_board, i, j + 1, check_types[type_c]
                        ):
                            legal_action[i][j] = 1
                            legal_action[i][j + 1] = 1
                            action_space[(self.num_col - 1) * i + j] = 1
                            # print((self.num_col - 1) * i + j)

                            match_4_v[i][j + 1] = 1
                            for x, y in check_types[type_c]:
                                match_4_v[i + x][j + 1 + y] = 1
                    for type_c in [15]:
                        if self.check_required_tile(
                                color, raw_board, i, j + 1, check_types[type_c]
                        ):
                            legal_action[i][j] = 1
                            legal_action[i][j + 1] = 1
                            action_space[(self.num_col - 1) * i + j] = 1
                            # print((self.num_col - 1) * i + j)

                            match_5[i][j + 1] = 1
                            for x, y in check_types[type_c]:
                                match_5[i + x][j + 1 + y] = 1
                    for type_c in [18, 19]:
                        if self.check_required_tile(
                                color, raw_board, i, j + 1, check_types[type_c]
                        ):
                            legal_action[i][j] = 1
                            legal_action[i][j + 1] = 1
                            action_space[(self.num_col - 1) * i + j] = 1
                            # print((self.num_col - 1) * i + j)

                            match_L[i][j + 1] = 1
                            for x, y in check_types[type_c]:
                                match_L[i + x][j + 1 + y] = 1
                    for type_c in [21]:
                        if self.check_required_tile(
                                color, raw_board, i, j + 1, check_types[type_c]
                        ):
                            legal_action[i][j] = 1
                            legal_action[i][j + 1] = 1
                            action_space[(self.num_col - 1) * i + j] = 1
                            # print((self.num_col - 1) * i + j)

                            match_T[i][j + 1] = 1
                            for x, y in check_types[type_c]:
                                match_T[i + x][j + 1 + y] = 1

            # wipe left
            if self.check_legal_pos_to_move(i, j - 1, raw_board):
                has_match_3 = False
                for type_c in [0, 1, 2, 5]:
                    if self.check_required_tile(
                            color, raw_board, i, j - 1, check_types[type_c]
                    ):
                        has_match_3 = True
                        legal_action[i][j] = 1
                        legal_action[i][j - 1] = 1
                        action_space[(self.num_col - 1) * i + (j - 1)] = 1
                        # print((self.num_col - 1) * i + (j - 1))

                        match_normal[i][j - 1] = 1
                        for x, y in check_types[type_c]:
                            match_normal[i + x][j - 1 + y] = 1
                for type_c in [6, 7]:
                    if self.check_required_tile(
                            color, raw_board, i, j - 1, check_types[type_c]
                    ):
                        legal_action[i][j] = 1
                        legal_action[i][j - 1] = 1
                        action_space[(self.num_col - 1) * i + (j - 1)] = 1
                        # print((self.num_col - 1) * i + (j - 1))

                        match_2x2[i][j - 1] = 1
                        for x, y in check_types[type_c]:
                            match_2x2[i + x][j - 1 + y] = 1
                        break
                if has_match_3:
                    for type_c in [12, 13]:
                        if self.check_required_tile(
                                color, raw_board, i, j - 1, check_types[type_c]
                        ):
                            legal_action[i][j] = 1
                            legal_action[i][j - 1] = 1
                            action_space[(self.num_col - 1) * i + (j - 1)] = 1
                            # print((self.num_col - 1) * i + (j - 1))

                            match_4_v[i][j - 1] = 1
                            for x, y in check_types[type_c]:
                                match_4_v[i + x][j - 1 + y] = 1
                    for type_c in [15]:
                        if self.check_required_tile(
                                color, raw_board, i, j - 1, check_types[type_c]
                        ):
                            legal_action[i][j] = 1
                            legal_action[i][j - 1] = 1
                            action_space[(self.num_col - 1) * i + (j - 1)] = 1
                            # print((self.num_col - 1) * i + (j - 1))

                            match_5[i][j - 1] = 1
                            for x, y in check_types[type_c]:
                                match_5[i + x][j - 1 + y] = 1
                    for type_c in [16, 17]:
                        if self.check_required_tile(
                                color, raw_board, i, j - 1, check_types[type_c]
                        ):
                            legal_action[i][j] = 1
                            legal_action[i][j - 1] = 1
                            action_space[(self.num_col - 1) * i + (j - 1)] = 1
                            # print((self.num_col - 1) * i + (j - 1))

                            match_L[i][j - 1] = 1
                            for x, y in check_types[type_c]:
                                match_L[i + x][j - 1 + y] = 1
                    for type_c in [23]:
                        if self.check_required_tile(
                                color, raw_board, i, j - 1, check_types[type_c]
                        ):
                            legal_action[i][j] = 1
                            legal_action[i][j - 1] = 1
                            action_space[(self.num_col - 1) * i + (j - 1)] = 1
                            # print((self.num_col - 1) * i + (j - 1))

                            match_T[i][j - 1] = 1
                            for x, y in check_types[type_c]:
                                match_T[i + x][j - 1 + y] = 1

            # wipe up
            if self.check_legal_pos_to_move(i - 1, j, raw_board):
                has_match_3 = False
                for type_c in [2, 3, 4, 5]:
                    if self.check_required_tile(
                            color, raw_board, i - 1, j, check_types[type_c]
                    ):
                        has_match_3 = True
                        legal_action[i][j] = 1
                        legal_action[i - 1][j] = 1
                        action_space[
                            (self.num_col - 1) * self.num_row
                            + self.num_col * (i - 1)
                            + j
                            ] = 1
                        # print((self.num_col - 1) * self.num_row + self.num_col * (i - 1) + j)

                        match_normal[i - 1][j] = 1
                        for x, y in check_types[type_c]:
                            match_normal[i - 1 + x][j + y] = 1
                for type_c in [6, 8]:
                    if self.check_required_tile(
                            color, raw_board, i - 1, j, check_types[type_c]
                    ):
                        legal_action[i][j] = 1
                        legal_action[i - 1][j] = 1
                        action_space[
                            (self.num_col - 1) * self.num_row
                            + self.num_col * (i - 1)
                            + j
                            ] = 1
                        # print((self.num_col - 1) * self.num_row + self.num_col * (i - 1) + j)

                        match_2x2[i - 1][j] = 1
                        for x, y in check_types[type_c]:
                            match_2x2[i - 1 + x][j + y] = 1
                        break
                if has_match_3:
                    for type_c in [10, 11]:
                        if self.check_required_tile(
                                color, raw_board, i - 1, j, check_types[type_c]
                        ):
                            legal_action[i][j] = 1
                            legal_action[i - 1][j] = 1
                            action_space[
                                (self.num_col - 1) * self.num_row
                                + self.num_col * (i - 1)
                                + j
                                ] = 1
                            # print((self.num_col - 1) * self.num_row + self.num_col * (i - 1) + j)

                            match_4_h[i - 1][j] = 1
                            for x, y in check_types[type_c]:
                                match_4_h[i - 1 + x][j + y] = 1
                    for type_c in [14]:
                        if self.check_required_tile(
                                color, raw_board, i - 1, j, check_types[type_c]
                        ):
                            legal_action[i][j] = 1
                            legal_action[i - 1][j] = 1
                            action_space[
                                (self.num_col - 1) * self.num_row
                                + self.num_col * (i - 1)
                                + j
                                ] = 1
                            # print((self.num_col - 1) * self.num_row + self.num_col * (i - 1) + j)

                            match_5[i - 1][j] = 1
                            for x, y in check_types[type_c]:
                                match_5[i - 1 + x][j + y] = 1
                    for type_c in [16, 19]:
                        if self.check_required_tile(
                                color, raw_board, i - 1, j, check_types[type_c]
                        ):
                            legal_action[i][j] = 1
                            legal_action[i - 1][j] = 1
                            action_space[
                                (self.num_col - 1) * self.num_row
                                + self.num_col * (i - 1)
                                + j
                                ] = 1
                            # print((self.num_col - 1) * self.num_row + self.num_col * (i - 1) + j)

                            match_L[i - 1][j] = 1
                            for x, y in check_types[type_c]:
                                match_L[i - 1 + x][j + y] = 1
                    for type_c in [20]:
                        if self.check_required_tile(
                                color, raw_board, i - 1, j, check_types[type_c]
                        ):
                            legal_action[i][j] = 1
                            legal_action[i - 1][j] = 1
                            action_space[
                                (self.num_col - 1) * self.num_row
                                + self.num_col * (i - 1)
                                + j
                                ] = 1
                            # print((self.num_col - 1) * self.num_row + self.num_col * (i - 1) + j)

                            match_T[i - 1][j] = 1
                            for x, y in check_types[type_c]:
                                match_T[i - 1 + x][j + y] = 1

            # wipe down
            if self.check_legal_pos_to_move(i + 1, j, raw_board):
                has_match_3 = False
                for type_c in [0, 3, 4, 5]:
                    if self.check_required_tile(
                            color, raw_board, i + 1, j, check_types[type_c]
                    ):
                        has_match_3 = True
                        legal_action[i][j] = 1
                        legal_action[i + 1][j] = 1
                        action_space[
                            (self.num_col - 1) * self.num_row + self.num_col * i + j
                            ] = 1
                        # print((self.num_col - 1) * self.num_row + self.num_col * i + j)

                        match_normal[i + 1][j] = 1
                        for x, y in check_types[type_c]:
                            match_normal[i + 1 + x][j + y] = 1
                for type_c in [7, 9]:
                    if self.check_required_tile(
                            color, raw_board, i + 1, j, check_types[type_c]
                    ):
                        legal_action[i][j] = 1
                        legal_action[i + 1][j] = 1
                        action_space[
                            (self.num_col - 1) * self.num_row + self.num_col * i + j
                            ] = 1
                        # print((self.num_col - 1) * self.num_row + self.num_col * i + j)

                        match_2x2[i + 1][j] = 1
                        for x, y in check_types[type_c]:
                            match_2x2[i + 1 + x][j + y] = 1
                        break
                if has_match_3:
                    for type_c in [10, 11]:
                        if self.check_required_tile(
                                color, raw_board, i + 1, j, check_types[type_c]
                        ):
                            legal_action[i][j] = 1
                            legal_action[i + 1][j] = 1
                            action_space[
                                (self.num_col - 1) * self.num_row + self.num_col * i + j
                                ] = 1
                            # print((self.num_col - 1) * self.num_row + self.num_col * i + j)

                            match_4_h[i + 1][j] = 1
                            for x, y in check_types[type_c]:
                                match_4_h[i + 1 + x][j + y] = 1
                    for type_c in [14]:
                        if self.check_required_tile(
                                color, raw_board, i + 1, j, check_types[type_c]
                        ):
                            legal_action[i][j] = 1
                            legal_action[i + 1][j] = 1
                            action_space[
                                (self.num_col - 1) * self.num_row + self.num_col * i + j
                                ] = 1
                            # print((self.num_col - 1) * self.num_row + self.num_col * i + j)

                            match_5[i + 1][j] = 1
                            for x, y in check_types[type_c]:
                                match_5[i + 1 + x][j + y] = 1
                    for type_c in [17, 18]:
                        if self.check_required_tile(
                                color, raw_board, i + 1, j, check_types[type_c]
                        ):
                            legal_action[i][j] = 1
                            legal_action[i + 1][j] = 1
                            action_space[
                                (self.num_col - 1) * self.num_row + self.num_col * i + j
                                ] = 1
                            # print((self.num_col - 1) * self.num_row + self.num_col * i + j)

                            match_L[i + 1][j] = 1
                            for x, y in check_types[type_c]:
                                match_L[i + 1 + x][j + y] = 1
                    for type_c in [22]:
                        if self.check_required_tile(
                                color, raw_board, i + 1, j, check_types[type_c]
                        ):
                            legal_action[i][j] = 1
                            legal_action[i + 1][j] = 1
                            action_space[
                                (self.num_col - 1) * self.num_row + self.num_col * i + j
                                ] = 1
                            # print((self.num_col - 1) * self.num_row + self.num_col * i + j)

                            match_T[i + 1][j] = 1
                            for x, y in check_types[type_c]:
                                match_T[i + 1 + x][j + y] = 1
            raw_board[i][j] = color

        return (
            match_normal,
            match_2x2,
            match_4_v,
            match_4_h,
            match_L,
            match_T,
            match_5,
            legal_action,
        )

    def _format_observation(
        self, board: np.array, list_monsters: list[AbstractMonster], device, time_factor: float
    ):
        """
        A utility function to process observations and move them to CUDA.
        """

        if board is None:
            return None

        if not device == "cpu":
            device = "cuda:" + str(device)

        action_space = np.zeros((self.num_action))
        obs = {
            "tiles": ((board == GameObject.immovable_shape) * 0
                    + (board == GameObject.color1) * 1
                    + (board == GameObject.color2) * 2
                    + (board == GameObject.color3) * 3
                    + (board == GameObject.color4) * 4
                    + (board == GameObject.color5) * 5
                    + (board == GameObject.power_disco) * 6
                    + (board == GameObject.power_bomb) * 7
                    + (board == GameObject.power_missile_h) * 8
                    + (board == GameObject.power_missile_v) * 9
                    + (board == GameObject.power_plane) * 10) / 10.0,
            "none_tile": (board == GameObject.immovable_shape),
            "color_1": (board == GameObject.color1),
            "color_2": (board == GameObject.color2),
            "color_3": (board == GameObject.color3),
            "color_4": (board == GameObject.color4),
            "color_5": (board == GameObject.color5),
            "disco": (board == GameObject.power_disco),
            "bomb": (board == GameObject.power_bomb),
            "missile_h": (board == GameObject.power_missile_h),
            "missile_v": (board == GameObject.power_missile_v),
            "plane": (board == GameObject.power_plane),
            # "buff": (board == GameObject.power_disco) \
            #         | (board == GameObject.power_disco) \
            #         | (board == GameObject.power_disco),
            "pu": (board == GameObject.power_disco)  \
                    + (board == GameObject.power_bomb)  \
                    + (board == GameObject.power_missile_h)  \
                    + (board == GameObject.power_missile_v) \
                    + (board == GameObject.power_plane) ,
            "blocker": (board == GameObject.blocker_box),
            # "monster": ((board == GameObject.monster_dame)
            # | (board == GameObject.monster_box_box)
            # | (board == GameObject.monster_box_bomb)
            # | (board == GameObject.monster_box_thorny)
            # | (board == GameObject.monster_box_both)
            # | (board == GameObject.blocker_thorny)
            # | (board == GameObject.blocker_bomb)),
            "monster": np.zeros((self.num_row, self .num_col)),
            "monster_match_dmg_mask": np.zeros((self.num_row, self.num_col)),
            "monster_match_hp": np.zeros((self.num_row, self.num_col)),
            "monster_inside_dmg_mask": np.zeros((self.num_row, self.num_col)),
            "monster_inside_hp": np.zeros((self.num_row, self.num_col)),
            "self_dmg_mask": np.zeros((self.num_row, self.num_col)),
            "match_normal": np.zeros((self.num_row, self.num_col)),
            "match_2x2": np.zeros((self.num_row, self.num_col)),
            "match_4_v": np.zeros((self.num_row, self.num_col)),
            "match_4_h": np.zeros((self.num_row, self.num_col)),
            "match_L": np.zeros((self.num_row, self.num_col)),
            "match_T": np.zeros((self.num_row, self.num_col)),
            "match_5": np.zeros((self.num_row, self.num_col)),
            "legal_action": np.zeros((self.num_row, self.num_col)),
            "heat_mask": np.full((self.num_row, self.num_col), time_factor),
        }


        for _mons in list_monsters:
            for p in _mons.mons_positions:
                try:
                    obs["monster"][p.get_coord()] = _mons.get_hp() / _mons._origin_hp
                except IndexError:
                    continue
            if isinstance(_mons, ThornyBlocker):
                for p in _mons.inside_dmg_mask:
                    try:
                        obs["self_dmg_mask"][p.get_coord()] = 1
                    except IndexError:
                        continue
            else:
                for p in _mons.inside_dmg_mask:
                    try:
                        obs["monster_inside_dmg_mask"][p.get_coord()] = 1
                        obs["monster_inside_hp"][p.get_coord()] = _mons.get_hp() / _mons._origin_hp
                    except IndexError:
                        continue

            for p in _mons.dmg_mask:
                try:
                    obs["monster_match_dmg_mask"][p.get_coord()] = 1
                    obs["monster_match_hp"][p.get_coord()] = _mons.get_hp() / _mons._origin_hp
                except IndexError:
                    continue

        for r in range(self.num_row):
            for c in range(self.num_col):
                tile = board[r][c]

                if tile in GameObject.set_powers_shape:
                    for i in [-1, 1]:
                        if self.check_legal_pos_to_move(r, c + i, board):
                            obs["legal_action"][r][c] = 1
                            obs["legal_action"][r][c + i] = 1
                            action_space[(self.num_col - 1) * r + (c + min(i, 0))] = 1
                        if self.check_legal_pos_to_move(r + i, c, board):
                            obs["legal_action"][r][c] = 1
                            obs["legal_action"][r + i][c] = 1
                            action_space[
                                (self.num_col - 1) * self.num_row
                                + self.num_col * (r + min(i, 0))
                                + c
                            ] = 1

        (
            obs["match_normal"],
            obs["match_2x2"],
            obs["match_4_v"],
            obs["match_4_h"],
            obs["match_L"],
            obs["match_T"],
            obs["match_5"],
            obs["legal_action"],
        ) = self.check_match(
            board,
            obs["match_normal"],
            obs["match_2x2"],
            obs["match_4_v"],
            obs["match_4_h"],
            obs["match_L"],
            obs["match_T"],
            obs["match_5"],
            obs["legal_action"],
            action_space,
        )

        return dict(obs=obs, action_space=action_space)

    def obs_to_tensor(self, obs):
        obs_tensor = torch.stack(([torch.Tensor(obs[attr]) for attr in self.obs_order]))
        return obs_tensor
