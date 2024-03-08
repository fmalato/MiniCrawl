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
    def __init__(self, starting_grid_size=3, max_grid_size=None, increment_freq=5, connection_density=0.5):
        assert 0 < connection_density <= 1, "connection_density must be in (0, 1]"
        assert starting_grid_size >= 2, "starting_grid_size must be in [2, +inf)"
        if max_grid_size is not None:
            assert max_grid_size > starting_grid_size, "max_grid_size must be greater than starting_grid_size"
        # Initialize attributes
        self._density = connection_density
        self._available_components = len(ObjectIndices)
        self._grid_size = starting_grid_size
        self._starting_grid_size = starting_grid_size
        self._max_grid_size = max_grid_size
        self._increment_freq = increment_freq
        self._grid = None
        self._grid_graphs = []
        self._connects = {}
        self._current_level = 0
        self._min_rooms = self._grid_size
        # Create first floor
        self._create_dungeon_floor()
        print("Hello")

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
        # Remove some rooms
        self._draw_rooms()
        corridor_trees = []
        self._connects = {}
        corridors = np.argwhere(self._grid == 0)
        for i, j in corridors:
            connections = self._get_connections(i, j)
            self._connects[(i, j)] = connections
            g = Graph(directed=False)
            g.add_node((i, j))
            for direction in connections:
                if direction == "east":
                    g.add_egde((i, j), (i, j + 1))
                elif direction == "west":
                    g.add_egde((i, j), (i, j - 1))
                elif direction == "north":
                    g.add_egde((i, j), (i - 1, j))
                else:
                    g.add_egde((i, j), (i + 1, j))
            if len(g.get_edges((i, j))) > 0:
                corridor_trees.append(g)

        self._build_maze_graph(corridor_trees)
        # Build a map for keeping track of node types
        for n in self._maze_graph.get_nodes():
            if self._grid[int(n[0]), int(n[1])] == 0:
                self._grid[int(n[0]), int(n[1])] = 2

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
        connections = list(neighbors.keys())
        to_remove = []
        # TODO: fix corridors logic
        for c in connections:
            if np.random.uniform(0, 1) <= 1 - self._density:
                to_remove.append(c)
                # General rule: if a previous neighbor (i.e. "north" or "west") have "south" or "east" connection, do not remove "north"/"west" from connections
                if c == "north" and row > 0 and self._grid[row - 1, col] == 0 and "south" in self._connects[(row - 1, col)]:
                    to_remove.remove(c)
                elif c == "west" and col > 0 and self._grid[row, col - 1] == 0 and "east" in self._connects[(row, col - 1)]:
                    to_remove.remove(c)

        for c in to_remove:
            connections.remove(c)

        """connections = []
        if np.random.uniform(0, 1) <= self._density:
            num_connections = np.random.randint(2, len(list(neighbors.keys())) + 1)
            connections = np.random.choice(list(neighbors.keys()), size=num_connections, replace=False)"""

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
            neighbors["north"] = self._grid[row - 1, col]
        if row < self._grid_size - 1:
            neighbors["south"] = self._grid[row + 1, col]
        if col > 0:
            neighbors["west"] = self._grid[row, col - 1]
        if col < self._grid_size - 1:
            neighbors["east"] = self._grid[row, col + 1]

        return neighbors

    def _build_maze_graph(self, corridor_trees):
        """
        Build a consistent maze graph for current floor, using a three-step process:
        - First step: build a graph using the corridor_trees. A corridor_tree is a tree with depth=1 where the root
                      is a corridor location with at least one connection to a  room, and each leaf is a connected
                      room.
        - Second step: adjust step 1 result by adding the terminal nodes
        - Third step: if there are multiple connected components, try to link them with an additional edge. The
                      procedure follows a heuristic, hence it might fail.
        :param corridor_trees: list of Graphs. Each graph is a tree with depth=1.
        :return: None
        """
        # Step 1: Build the initial graph
        self._maze_graph = Graph(directed=False)
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

        # Step 2: Add terminal nodes
        for n in self._maze_graph.get_nodes():
            for e in self._maze_graph.get_edges(n):
                if e not in self._maze_graph.get_nodes():
                    self._maze_graph.add_node(e)
        # Step 3: Compute connected components, in case there's more than one
        connected_components = self._maze_graph.connected_components()
        # Heuristic: if more than one component, try to connect them
        if len(connected_components) > 1:
            biggest = connected_components[np.argmax([len(x) for x in connected_components])]
            connected_components.remove(biggest)
            for comp in connected_components:
                self._maze_graph.connect_components(biggest, comp)

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