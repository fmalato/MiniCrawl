import numpy as np

from minicrawl.graph import Graph
from minicrawl.utils import minimum_spanning_tree


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
    def __init__(self, starting_grid_size=3, max_grid_size=None, increment_freq=5, connection_density=0.5):
        assert 0 < connection_density <= 1, "connection_density must be in (0, 1]"
        assert starting_grid_size >= 2, "starting_grid_size must be in [2, +inf)"
        if max_grid_size is not None:
            assert max_grid_size > starting_grid_size, "max_grid_size must be greater than starting_grid_size"
        # Initialize attributes
        self._density = connection_density
        self._grid_size = starting_grid_size
        self._starting_grid_size = starting_grid_size
        self._max_grid_size = max_grid_size
        self._increment_freq = increment_freq
        self._grid = None
        self._grid_graphs = []
        self._connects = {}
        self._current_level = 0
        self._min_rooms = self._grid_size

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
        # Remove some rooms
        self._draw_rooms()
        connects = {}
        for i, j in np.ndindex(self._grid.shape):
            connections = self._get_neighbors(i, j)
            connects[(i, j)] = connections
        self._build_maze_graph(connects)
        # Build a map for keeping track of node types
        for n in self._maze_graph.get_nodes():
            if self._grid[int(n[0]), int(n[1])] == 0:
                self._grid[int(n[0]), int(n[1])] = 2
        # Build connection map
        for i, j in np.ndindex(self._grid.shape):
            self._get_connections(i, j)

    def _draw_rooms(self):
        rooms = np.argwhere(self._grid == 1)
        num_rooms = int(np.sum(self._grid))
        for i, j in rooms:
            obj = np.random.randint(0, 2)
            if obj == 0 and num_rooms > self._min_rooms:
                num_rooms -= 1
                self._grid[i, j] = obj

    def _get_connections(self, row, col):
        assert self._maze_graph is not None, "_get_connections() must be called after the floor graph is created."
        self._connects[(row, col)] = {}
        for e in self._maze_graph.get_edges((row, col)):
            if e == (row, col + 1):
                self._connects[(row, col)]["east"] = self._grid[row, col + 1]
            elif e == (row, col - 1):
                self._connects[(row, col)]["west"] = self._grid[row, col - 1]
            elif e == (row - 1, col):
                self._connects[(row, col)]["north"] = self._grid[row - 1, col]
            elif e == (row + 1, col):
                self._connects[(row, col)]["south"] = self._grid[row + 1, col]

    def _get_neighbors(self, row, col):
        """
            Retrieves objects in Manhattan neighborhood
        :param row: int - row index
        :param col: int - col index
        :return: dict - manhattan-neighbors of current location
        """
        neighbors = {}
        if row > 0:
            neighbors["north"] = self._grid[row - 1, col]
        if row < self._grid_size - 1:
            neighbors["south"] = self._grid[row + 1, col]
        if col > 0:
            neighbors["west"] = self._grid[row, col - 1]
        if col < self._grid_size - 1:
            neighbors["east"] = self._grid[row, col + 1]

        return neighbors

    def _build_maze_graph(self, connects):
        # Step 1: Build graph with all connections (very dense, in some cases might be complete)
        g = Graph(directed=False)
        for i, j in np.ndindex(self._grid.shape):
            try:
                if (i, j) not in g.get_nodes():
                    g.add_node((i, j))
                for k in connects[(i, j)].keys():
                    if k == "east":
                        g.add_egde((i, j), (i, j + 1))
                    elif k == "west":
                        g.add_egde((i, j), (i, j - 1))
                    elif k == "north":
                        g.add_egde((i, j), (i - 1, j))
                    else:
                        g.add_egde((i, j), (i + 1, j))
            except KeyError:
                continue
        # Step 2: build minimum spanning tree starting from a random node
        nodes = g.get_nodes()
        node_idx = np.random.randint(0, len(nodes))
        self._maze_graph = minimum_spanning_tree(g, nodes[node_idx])
        # Step 3: build some edges back
        num_back_edges = np.random.randint(0, 10)
        for i in range(num_back_edges):
            node_idx = np.random.choice(len(nodes))
            n = g.get_nodes()[node_idx]
            neighbors = self._get_neighbors(n[0], n[1])
            edge_idx = np.random.choice(len(neighbors))
            k = list(neighbors.keys())[edge_idx]
            if k == "east":
                e = (n[0], n[1] + 1)
            elif k == "west":
                e = (n[0], n[1] - 1)
            elif k == "north":
                e = (n[0] - 1, n[1])
            else:
                e = (n[0] + 1, n[1])
            self._maze_graph.add_egde(n, e)

    def get_current_level(self):
        return self._current_level

    def increment_level(self):
        self._current_level += 1
        self._create_dungeon_floor()

    def reset(self):
        self._current_level = 0
        self._grid_size = self._starting_grid_size
        self._create_dungeon_floor()

    def get_grid_size(self):
        return (self._grid_size, self._grid_size)

    def increment_grid_size(self):
        if self._max_grid_size is not None:
            self._grid_size += 1
        else:
            self._grid_size = np.min(self._grid_size + 1, self._max_grid_size)

    def get_current_floor(self):
        return self._maze_graph, self._grid

    def get_connections(self):
        return self._connects

    def get_connections_for_room(self, position):
        return self._connects[position]
