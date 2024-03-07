import numpy as np

from minicrawl.params import ObjectIndices
from minicrawl.graph import Graph


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
    def __init__(self, grid_size, starting_grid_size=3, max_grid_size=None, increment_freq=5, connection_density=0.5):
        assert 0 < connection_density <= 1, "connection_density must be in (0, 1]"
        assert starting_grid_size >= 2, "starting_grid_size must be in [2, +inf)"
        if max_grid_size is not None:
            assert max_grid_size > starting_grid_size, "max_grid_size must be greater than starting_grid_size"
        # Initialize attributes
        self._density = connection_density
        self._available_components = len(ObjectIndices)
        self._grid_size = grid_size
        self._starting_grid_size = starting_grid_size
        self._max_grid_size = max_grid_size
        self._increment_freq = increment_freq
        self._grid = None
        self._grid_graphs = []
        self._room_connections = {}
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
        for coords in np.argwhere(self._grid == 1):
            self._room_connections[tuple(coords)] = []
        # TODO: implement later on
        """# Remove some rooms
        self._draw_rooms()"""
        # TODO: creation rules
        corridor_trees = []
        connects = {}
        for i in range(current_grid_size):
            for j in range(current_grid_size):
                if self._grid[i, j] != 1:
                    connections = self._get_connections(i, j)
                    connects[(i, j)] = connections
                    g = Graph(directed=True)
                    g.add_node((i, j))
                    for direction in connections:
                        if direction == "right":
                            g.add_egde((i, j), (i, j + 1))
                        elif direction == "left":
                            g.add_egde((i, j), (i, j - 1))
                        elif direction == "top":
                            g.add_egde((i, j), (i - 1, j))
                        else:
                            g.add_egde((i, j), (i + 1, j))
                    if len(g.get_edges((i, j))) > 0:
                        corridor_trees.append(g)

        self._build_maze_graph(corridor_trees)
        #max_distance = self._maze_graph.longest_walkable_path()
        #connected_components, depth, max_depth = self._maze_graph.dfs(self._maze_graph.get_nodes()[0], depth=0)

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
        if np.random.uniform(0, 1) <= self._density:
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

    def _build_maze_graph(self, corridor_trees):
        self._maze_graph = Graph(directed=True)
        for g in corridor_trees:
            tmp = [x for x in corridor_trees if x != g]
            # At this stage, there's only one node
            node = g.get_nodes()[0]
            for edge in g.get_edges(node):
                if node not in self._maze_graph.get_nodes():
                    self._maze_graph.add_node(node)
                self._maze_graph.add_egde(node, edge)
                for t in tmp:
                    n = t.get_nodes()[0]
                    if edge in t.get_edges(n):
                        if edge not in self._maze_graph.get_nodes():
                            self._maze_graph.add_node(edge)
                        self._maze_graph.add_egde(edge, n)
                        for x in corridor_trees:
                            if x == t:
                                x.remove_edge(n, edge)

    def increment_level(self):
        self._current_level += 1
        self._create_dungeon_floor()
