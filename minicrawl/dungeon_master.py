import numpy as np

from params import ObjectIndices


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
        # Create first floor
        self._create_dungeon_floor()


    def _create_dungeon_floor(self):
        """
            Creates a random floor plan of size (self._grid_size, self._grid_size)
        :return: None
        """
        current_grid_size = self._grid_size + int(self._current_level / self._increment_freq)
        self._grid = (np.resize([1, 0], current_grid_size ** 2))
        self._grid = np.reshape(a=self._grid, newshape=(current_grid_size, current_grid_size))
        # TODO: creation rules
        for i in range(current_grid_size):
            for j in range(current_grid_size):
                possible_components = np.arange(self._available_components)
                neighbors = self._get_neighbors(i, j)
                if neighbors["top"] == ObjectIndices.ROOM or neighbors["bottom"] == ObjectIndices.ROOM or neighbors["left"] == ObjectIndices.ROOM or neighbors["right"] == ObjectIndices.ROOM:
                    possible_components = np.delete(possible_components, ObjectIndices.ROOM)

    def _get_neighbors(self, i, j):
        """
            Retrieves objects in Manhattan neighborhood
        :param i: int - row index
        :param j: int - col index
        :return: dict - manhattan-neighbors of current location
        """
        neighbors = {}
        if i > 0:
            neighbors["top"] = self._grid[i - 1, j]
        if i < self._grid_size - 1:
            neighbors["bottom"] = self._grid[i + 1, j]
        if j > 0:
            neighbors["left"] = self._grid[i, j - 1]
        if j > self._grid_size - 1:
            neighbors["right"] = self._grid[i, j + 1]

        return neighbors

    def increment_level(self):
        self._current_level += 1
        self._create_dungeon_floor()