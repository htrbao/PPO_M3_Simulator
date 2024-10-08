# cython: boundscheck=False, wraparound=False, nonecheck=False
# cython: profile=True
# cython: language_level=3

import numpy as np
cimport numpy as np

# -------- GAME - FUNCTIONS --------
cdef bint is_valid_point(int row, int col, int max_row, int max_col):
    return 0 <= row < max_row and 0 <= col < max_col

cpdef generator_neighbours(
        int board_rows, int board_cols,
        np.ndarray[np.float64_t, ndim=2] board_contain_shapes,
        int curRow,
        int curCol,
        float filter_shape,
        list search_directions,
        bint early_stop=False):
    cdef:
        list lst_cells = []
        int idx, newRow, newCol
        float shape
        list newCells
        list axis_dirs
        list dir_
    for idx, axis_dirs in enumerate(search_directions):
        newCells = []
        for dir_ in axis_dirs:
            newRow, newCol = curRow + dir_[0], curCol + dir_[1]
            if not is_valid_point(newRow, newCol, board_rows, board_cols):
                lst_cells.append(([], 0, -1))
                break

            shape = board_contain_shapes[newRow, newCol]
            if shape != filter_shape:
                continue

            newCells.append((shape, newRow, newCol))
        else:
            lst_cells.append((newCells, len(axis_dirs), idx))

        if early_stop:
            break
    return lst_cells

# -------------------------------------------------------------------------------------------------------------------
# -------- MATCH3 HELPER - FUNCTIONS --------
cdef tuple[] check_types = (
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

cdef bint check_legal_pos_to_move(int row, int col, int max_row, int max_col,
                                  np.ndarray[np.float64_t, ndim=2] raw_board,
                                  set set_movable_shape):
    return 0 <= row < max_row and 0 <= col < max_col and raw_board[row, col] in set_movable_shape

cdef check_required_tile(
        np.ndarray[np.uint8_t, ndim=2, cast=True] color_board,
        np.ndarray[np.float64_t, ndim=2] raw_board,
        int i, int j,
        int max_row, int max_col,
        int idx_check_tile,
        set set_movable_shape
):
    for x, y in check_types[idx_check_tile]:
        if (
                not check_legal_pos_to_move(i + x, j + y, max_row, max_col, raw_board, set_movable_shape)
                or color_board[i + x, j + y] != 1
        ):
            return False

    return True

cpdef check_match(
        np.ndarray[np.float64_t, ndim=2] raw_board,
        np.ndarray[np.uint8_t, ndim=2, cast=True] color_board,
        np.ndarray[np.float64_t, ndim=2] match_normal,
        np.ndarray[np.float64_t, ndim=2] match_2x2,
        np.ndarray[np.float64_t, ndim=2] match_4_v,
        np.ndarray[np.float64_t, ndim=2] match_4_h,
        np.ndarray[np.float64_t, ndim=2] match_L,
        np.ndarray[np.float64_t, ndim=2] match_T,
        np.ndarray[np.float64_t, ndim=2] match_5,
        np.ndarray[np.float64_t, ndim=2] legal_action,
        np.ndarray[np.float64_t, ndim=1] action_space,
        int max_row, int max_col,
        set set_movable_shape
):
    for i in range(max_row):
        for j in range(max_col):
            if not color_board[i, j] == 1:
                continue
            # wipe right
            color_board[i, j] = 255
            if check_legal_pos_to_move(i, j + 1, max_row, max_col, raw_board, set_movable_shape):
                has_match_3 = False
                for type_c in [0, 1, 2, 3]:
                    if check_required_tile(
                            color_board, raw_board, i, j + 1, max_row, max_col, type_c, set_movable_shape
                    ):
                        has_match_3 = True
                        legal_action[i, j] = 1
                        legal_action[i, j + 1] = 1
                        action_space[(max_col - 1) * i + j] = 1

                        match_normal[i, j + 1] = 1
                        for x, y in check_types[type_c]:
                            match_normal[i + x, j + 1 + y] = 1
                for type_c in [8, 9]:
                    if check_required_tile(
                            color_board, raw_board, i, j + 1, max_row, max_col, type_c, set_movable_shape
                    ):
                        legal_action[i, j] = 1
                        legal_action[i, j + 1] = 1
                        action_space[(max_col - 1) * i + j] = 1

                        match_2x2[i, j + 1] = 1
                        for x, y in check_types[type_c]:
                            match_2x2[i + x, j + 1 + y] = 1
                        break
                if has_match_3:
                    for type_c in [12, 13]:
                        if check_required_tile(
                                color_board, raw_board, i, j + 1, max_row, max_col, type_c, set_movable_shape
                        ):
                            legal_action[i, j] = 1
                            legal_action[i, j + 1] = 1
                            action_space[(max_col - 1) * i + j] = 1

                            match_4_v[i, j + 1] = 1
                            for x, y in check_types[type_c]:
                                match_4_v[i + x, j + 1 + y] = 1
                    for type_c in [15]:
                        if check_required_tile(
                                color_board, raw_board, i, j + 1, max_row, max_col, type_c, set_movable_shape
                        ):
                            legal_action[i, j] = 1
                            legal_action[i, j + 1] = 1
                            action_space[(max_col - 1) * i + j] = 1

                            match_5[i, j + 1] = 1
                            for x, y in check_types[type_c]:
                                match_5[i + x, j + 1 + y] = 1
                    for type_c in [18, 19]:
                        if check_required_tile(
                                color_board, raw_board, i, j + 1, max_row, max_col, type_c, set_movable_shape
                        ):
                            legal_action[i, j] = 1
                            legal_action[i, j + 1] = 1
                            action_space[(max_col - 1) * i + j] = 1

                            match_L[i, j + 1] = 1
                            for x, y in check_types[type_c]:
                                match_L[i + x, j + 1 + y] = 1
                    for type_c in [21]:
                        if check_required_tile(
                                color_board, raw_board, i, j + 1, max_row, max_col, type_c, set_movable_shape
                        ):
                            legal_action[i, j] = 1
                            legal_action[i, j + 1] = 1
                            action_space[(max_col - 1) * i + j] = 1

                            match_T[i, j + 1] = 1
                            for x, y in check_types[type_c]:
                                match_T[i + x, j + 1 + y] = 1

            # wipe left
            if check_legal_pos_to_move(i, j - 1, max_row, max_col, raw_board, set_movable_shape):
                has_match_3 = False
                for type_c in [0, 1, 2, 5]:
                    if check_required_tile(
                            color_board, raw_board, i, j - 1, max_row, max_col, type_c, set_movable_shape
                    ):
                        has_match_3 = True
                        legal_action[i, j] = 1
                        legal_action[i, j - 1] = 1
                        action_space[(max_col - 1) * i + (j - 1)] = 1

                        match_normal[i, j - 1] = 1
                        for x, y in check_types[type_c]:
                            match_normal[i + x, j - 1 + y] = 1
                for type_c in [6, 7]:
                    if check_required_tile(
                            color_board, raw_board, i, j - 1, max_row, max_col, type_c, set_movable_shape
                    ):
                        legal_action[i, j] = 1
                        legal_action[i, j - 1] = 1
                        action_space[(max_col - 1) * i + (j - 1)] = 1

                        match_2x2[i, j - 1] = 1
                        for x, y in check_types[type_c]:
                            match_2x2[i + x, j - 1 + y] = 1
                        break
                if has_match_3:
                    for type_c in [12, 13]:
                        if check_required_tile(
                                color_board, raw_board, i, j - 1, max_row, max_col, type_c, set_movable_shape
                        ):
                            legal_action[i, j] = 1
                            legal_action[i, j - 1] = 1
                            action_space[(max_col - 1) * i + (j - 1)] = 1

                            match_4_v[i, j - 1] = 1
                            for x, y in check_types[type_c]:
                                match_4_v[i + x, j - 1 + y] = 1
                    for type_c in [15]:
                        if check_required_tile(
                                color_board, raw_board, i, j - 1, max_row, max_col, type_c, set_movable_shape
                        ):
                            legal_action[i, j] = 1
                            legal_action[i, j - 1] = 1
                            action_space[(max_col - 1) * i + (j - 1)] = 1

                            match_5[i, j - 1] = 1
                            for x, y in check_types[type_c]:
                                match_5[i + x, j - 1 + y] = 1
                    for type_c in [16, 17]:
                        if check_required_tile(
                                color_board, raw_board, i, j - 1, max_row, max_col, type_c, set_movable_shape
                        ):
                            legal_action[i, j] = 1
                            legal_action[i, j - 1] = 1
                            action_space[(max_col - 1) * i + (j - 1)] = 1

                            match_L[i, j - 1] = 1
                            for x, y in check_types[type_c]:
                                match_L[i + x, j - 1 + y] = 1
                    for type_c in [23]:
                        if check_required_tile(
                                color_board, raw_board, i, j - 1, max_row, max_col, type_c, set_movable_shape
                        ):
                            legal_action[i, j] = 1
                            legal_action[i, j - 1] = 1
                            action_space[(max_col - 1) * i + (j - 1)] = 1

                            match_T[i, j - 1] = 1
                            for x, y in check_types[type_c]:
                                match_T[i + x, j - 1 + y] = 1

            # wipe up
            if check_legal_pos_to_move(i - 1, j, max_row, max_col, raw_board, set_movable_shape):
                has_match_3 = False
                for type_c in [2, 3, 4, 5]:
                    if check_required_tile(
                            color_board, raw_board, i - 1, j, max_row, max_col, type_c, set_movable_shape
                    ):
                        has_match_3 = True
                        legal_action[i, j] = 1
                        legal_action[i - 1, j] = 1
                        action_space[
                            (max_col - 1) * max_row
                            + max_col * (i - 1)
                            + j
                            ] = 1

                        match_normal[i - 1, j] = 1
                        for x, y in check_types[type_c]:
                            match_normal[i - 1 + x, j + y] = 1
                for type_c in [6, 8]:
                    if check_required_tile(
                            color_board, raw_board, i - 1, j, max_row, max_col, type_c, set_movable_shape
                    ):
                        legal_action[i, j] = 1
                        legal_action[i - 1, j] = 1
                        action_space[
                            (max_col - 1) * max_row
                            + max_col * (i - 1)
                            + j
                            ] = 1

                        match_2x2[i - 1, j] = 1
                        for x, y in check_types[type_c]:
                            match_2x2[i - 1 + x, j + y] = 1
                        break
                if has_match_3:
                    for type_c in [10, 11]:
                        if check_required_tile(
                                color_board, raw_board, i - 1, j, max_row, max_col, type_c, set_movable_shape
                        ):
                            legal_action[i, j] = 1
                            legal_action[i - 1, j] = 1
                            action_space[
                                (max_col - 1) * max_row
                                + max_col * (i - 1)
                                + j
                                ] = 1

                            match_4_h[i - 1, j] = 1
                            for x, y in check_types[type_c]:
                                match_4_h[i - 1 + x, j + y] = 1
                    for type_c in [14]:
                        if check_required_tile(
                                color_board, raw_board, i - 1, j, max_row, max_col, type_c, set_movable_shape
                        ):
                            legal_action[i, j] = 1
                            legal_action[i - 1, j] = 1
                            action_space[
                                (max_col - 1) * max_row
                                + max_col * (i - 1)
                                + j
                                ] = 1

                            match_5[i - 1, j] = 1
                            for x, y in check_types[type_c]:
                                match_5[i - 1 + x, j + y] = 1
                    for type_c in [16, 19]:
                        if check_required_tile(
                                color_board, raw_board, i - 1, j, max_row, max_col, type_c, set_movable_shape
                        ):
                            legal_action[i, j] = 1
                            legal_action[i - 1, j] = 1
                            action_space[
                                (max_col - 1) * max_row
                                + max_col * (i - 1)
                                + j
                                ] = 1

                            match_L[i - 1, j] = 1
                            for x, y in check_types[type_c]:
                                match_L[i - 1 + x, j + y] = 1
                    for type_c in [20]:
                        if check_required_tile(
                                color_board, raw_board, i - 1, j, max_row, max_col, type_c, set_movable_shape
                        ):
                            legal_action[i, j] = 1
                            legal_action[i - 1, j] = 1
                            action_space[
                                (max_col - 1) * max_row
                                + max_col * (i - 1)
                                + j
                                ] = 1

                            match_T[i - 1, j] = 1
                            for x, y in check_types[type_c]:
                                match_T[i - 1 + x, j + y] = 1

            # wipe down
            if check_legal_pos_to_move(i + 1, j, max_row, max_col, raw_board, set_movable_shape):
                has_match_3 = False
                for type_c in [0, 3, 4, 5]:
                    if check_required_tile(
                            color_board, raw_board, i + 1, j, max_row, max_col, type_c, set_movable_shape
                    ):
                        has_match_3 = True
                        legal_action[i, j] = 1
                        legal_action[i + 1, j] = 1
                        action_space[
                            (max_col - 1) * max_row + max_col * i + j
                            ] = 1

                        match_normal[i + 1, j] = 1
                        for x, y in check_types[type_c]:
                            match_normal[i + 1 + x, j + y] = 1
                for type_c in [7, 9]:
                    if check_required_tile(
                            color_board, raw_board, i + 1, j, max_row, max_col, type_c, set_movable_shape
                    ):
                        legal_action[i, j] = 1
                        legal_action[i + 1, j] = 1
                        action_space[
                            (max_col - 1) * max_row + max_col * i + j
                            ] = 1

                        match_2x2[i + 1, j] = 1
                        for x, y in check_types[type_c]:
                            match_2x2[i + 1 + x, j + y] = 1
                        break
                if has_match_3:
                    for type_c in [10, 11]:
                        if check_required_tile(
                                color_board, raw_board, i + 1, j, max_row, max_col, type_c, set_movable_shape
                        ):
                            legal_action[i, j] = 1
                            legal_action[i + 1, j] = 1
                            action_space[
                                (max_col - 1) * max_row + max_col * i + j
                                ] = 1

                            match_4_h[i + 1, j] = 1
                            for x, y in check_types[type_c]:
                                match_4_h[i + 1 + x, j + y] = 1
                    for type_c in [14]:
                        if check_required_tile(
                                color_board, raw_board, i + 1, j, max_row, max_col, type_c, set_movable_shape
                        ):
                            legal_action[i, j] = 1
                            legal_action[i + 1, j] = 1
                            action_space[
                                (max_col - 1) * max_row + max_col * i + j
                                ] = 1

                            match_5[i + 1, j] = 1
                            for x, y in check_types[type_c]:
                                match_5[i + 1 + x, j + y] = 1
                    for type_c in [17, 18]:
                        if check_required_tile(
                                color_board, raw_board, i + 1, j, max_row, max_col, type_c, set_movable_shape
                        ):
                            legal_action[i, j] = 1
                            legal_action[i + 1, j] = 1
                            action_space[
                                (max_col - 1) * max_row + max_col * i + j
                                ] = 1

                            match_L[i + 1, j] = 1
                            for x, y in check_types[type_c]:
                                match_L[i + 1 + x, j + y] = 1
                    for type_c in [22]:
                        if check_required_tile(
                                color_board, raw_board, i + 1, j, max_row, max_col, type_c, set_movable_shape
                        ):
                            legal_action[i, j] = 1
                            legal_action[i + 1, j] = 1
                            action_space[
                                (max_col - 1) * max_row + max_col * i + j
                                ] = 1

                            match_T[i + 1, j] = 1
                            for x, y in check_types[type_c]:
                                match_T[i + 1 + x, j + y] = 1
            color_board[i, j] = 1

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
