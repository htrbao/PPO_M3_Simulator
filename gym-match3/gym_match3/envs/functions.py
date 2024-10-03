from collections import deque


def is_valid_point(row, col, board_rows, board_cols):
    return (((row + 1) <= board_rows) and (row >= 0)) and (((col + 1) <= board_cols) and (col >= 0))


class CellPoolManager:
    """Singleton class to manage a pool of Point objects."""
    _instance = None
    _pool = None
    _max_size = 100  # Default maximum size of the pool

    def __new__(cls, max_size=None):
        if cls._instance is None:
            cls._instance = super(CellPoolManager, cls).__new__(cls)
            cls._pool = deque()
            if max_size is not None:
                cls._max_size = max_size
            cls._instance.init_first()
        return cls._instance

    def init_first(self):
        from gym_match3.envs.game import Cell
        for _ in range(self._max_size):
            self._pool.append(Cell(-1, 0, 0))

    def get_cell(self, shape, row, col):
        # Check for available point in the pool
        if self._pool:
            p = self._pool.popleft()
            p.set_cell(shape, row, col)
            return p
        else:
            from gym_match3.envs.game import Cell
            return Cell(shape, row, col)

    def return_cell(self, point):
        if len(self._pool) < self._max_size:
            self._pool.append(point)  # Add back to the pool for reuse

    def __str__(self):
        return f"PointPoolManager with pool size: {len(self._pool)}"
