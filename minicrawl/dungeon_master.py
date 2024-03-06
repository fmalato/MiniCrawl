import numpy as np

from minicrawl.params import ObjectIndices


class DungeonMaster:
    """
        A Dungeon Master creates the dungeon and populates it with the challenges.
        The dungeon is created as a grid of int, where each value corresponds to a specific piece.
        Values:
        0 - Empty
        1 - Rect Room
        2 - Corridor
        3 - Angle junction
        4 - T-junction
    """
    def __init__(self, grid_size, starting_grid_size=3, max_grid_size=-1, increment_freq=5):
        # Initialize attributes
        self._available_components = len(ObjectIndices)
        self._grid_size = grid_size
        self._starting_grid_size = starting_grid_size
        self._max_grid_size = max_grid_size
        self._increment_freq = increment_freq
        self._grid = None
        self._current_level = 0
        self._min_rooms = 4
        # Create first floor
        self._create_dungeon_floor()

    def _create_dungeon_floor(self):
        """
            Creates a random floor plan of size (self._grid_size, self._grid_size)
        :return: None
        """
        current_grid_size = self._grid_size + int(self._current_level / self._increment_freq)
        self._grid = np.zeros(shape=(current_grid_size, current_grid_size))
        self._grid[::2, ::2] = 1
        self._grid[1::2, 1::2] = 1
        # TODO: implement later on
        """# Remove some rooms
        self._draw_rooms()"""
        # TODO: creation rules
        corridors = {}
        for i in range(current_grid_size):
            for j in range(current_grid_size):
                if self._grid[i, j] != 1:
                    connections = self._get_connections(i, j)
                    #self._grid[i, j] = connections
                    corridors[(i, j)] = connections

        print("Hello")

    def _draw_rooms(self):
        num_rooms = int(np.sum(self._grid))
        for i, j in np.ndindex(self._grid.shape):
            if self._grid[i, j] == 1:
                obj = np.random.randint(0, 2)
                if obj == 0 and num_rooms > self._min_rooms:
                    num_rooms -= 1
                    self._grid[i, j] = obj

    def _get_connections(self, row, col):
        neighbors = self._get_neighbors(row, col)
        connections = []
        if np.random.uniform(0, 1) >= 0.5:
            num_connections = np.random.randint(2, len(list(neighbors.keys())) + 1)
            connections = np.random.choice(list(neighbors.keys()), size=num_connections, replace=False)

        return connections



    def _get_neighbors(self, row, col):
        """
            Retrieves objects in Manhattan neighborhood
        :param row: int - row index
        :param col: int - col index
        :return: dict - manhattan-neighbors of current location
        """
        neighbors = {}
        if row > 0:
            neighbors["top"] = self._grid[row - 1, col]
        if row < self._grid_size - 1:
            neighbors["bottom"] = self._grid[row + 1, col]
        if col > 0:
            neighbors["left"] = self._grid[row, col - 1]
        if col < self._grid_size - 1:
            neighbors["right"] = self._grid[row, col + 1]

        return neighbors

    def increment_level(self):
        self._current_level += 1
        self._create_dungeon_floor()
