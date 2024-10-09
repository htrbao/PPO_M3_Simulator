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
                break

            newCells.append((shape, newRow, newCol))
        else:
            lst_cells.append((newCells, len(axis_dirs), idx))

        if early_stop:
            break
    return lst_cells

# -------------------------------------------------------------------------------------------------------------------