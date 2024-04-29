from typing import Optional

import numpy as np
import gymnasium as gym

from minicrawl.graph import Graph
from minicrawl.utils import minimum_spanning_tree
from minicrawl.params import COLORS, BOSS_STAGES


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
        self._current_level = 1
        self._min_rooms = self._grid_size
        self._floor_map_edge = 80

        self._create_dungeon_floor()
        self.level_name = "dungeon_floor"

    def _create_boss_stage(self, stage_type):
        self._grid = np.zeros(shape=(1, 1))
        self._grid[0, 0] = 1
        self._maze_graph = Graph(directed=False)
        self._maze_graph.add_node((0, 0))
        self._connects[(0, 0)] = {}

    def _create_dungeon_floor(self):
        """
            Creates a random floor plan of size (self._grid_size, self._grid_size)
        :return: None
        """
        self._grid = np.zeros(shape=(self._grid_size, self._grid_size))
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

    def create_dungeon_floor(self):
        self._create_dungeon_floor()

        return self._maze_graph, self._grid, self._connects

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

        if self._current_level % self._increment_freq == 0 and self._current_level != 0:
            boss_stage_type = np.random.choice(BOSS_STAGES)
            self._create_boss_stage(stage_type=boss_stage_type)
            self.level_name = f"{boss_stage_type}_boss_stage"
        else:
            if self._current_level % self._increment_freq == 1 and self._current_level != 1:
                self.increment_grid_size()
            self._create_dungeon_floor()
            self.level_name = "dungeon_floor"

        return self.level_name

    def increment_level_count(self):
        self._current_level += 1

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

    def get_current_level_name(self):
        return self.level_name

    def get_connections(self):
        return self._connects

    def get_connections_for_room(self, position):
        return self._connects[position]

    def choose_goal_and_agent_positions(self):
        if self._current_level % self._increment_freq == 0 and self._current_level != 0:
            return (0, 0), (0, 0)
        else:
            rooms_names = list(np.argwhere(self._grid == 1))
            stairs_room_idx = np.random.randint(0, len(rooms_names))
            stairs_room = tuple(rooms_names[stairs_room_idx])
            # Ensure exploration by placing agent in a different room
            rooms_names.pop(stairs_room_idx)
            agent_room_idx = np.random.randint(0, len(rooms_names))
            agent_room = tuple(rooms_names[agent_room_idx])

        return stairs_room, agent_room

    def choose_goal_position(self):
        if self._current_level % self._increment_freq == 0 and self._current_level != 0:
            return (0, 0)
        else:
            rooms_names = list(np.argwhere(self._grid == 1))
            stairs_room_idx = np.random.randint(0, len(rooms_names))
            stairs_room = tuple(rooms_names[stairs_room_idx])

        return stairs_room

    def choose_agent_position(self, level_name):
        if level_name != "dungeon_floor":
            return (0, 0)
        rooms_names = list(np.argwhere(self._grid == 1))
        agent_room_idx = np.random.randint(0, len(rooms_names))
        agent_room = tuple(rooms_names[agent_room_idx])

        return agent_room

    def build_floor_map(self, agent_pos=None, agent_dir=None, goal_pos=None, cell_size=9):
        # TODO: grid_size >= 9 is confusing (workaround: grid 9x9 never reached for now)
        cell_px_size = int(self._floor_map_edge / self._grid_size)
        floor_map = np.zeros(shape=(self._floor_map_edge, self._floor_map_edge, 3), dtype=np.uint8)
        # Draw corridors
        half_cell = int(cell_px_size / 2)
        quarter_cell = int(cell_px_size / 4)
        corridors = np.argwhere(self._grid == 2)
        for c in corridors:
            for conn in self._connects[tuple(c)].keys():
                if conn == "north":
                    start_x = c[0] * cell_px_size
                    end_x = c[0] * cell_px_size + half_cell
                    start_y = c[1] * cell_px_size + quarter_cell
                    end_y = (c[1] + 1) * cell_px_size - quarter_cell
                elif conn == "east":
                    start_x = c[0] * cell_px_size + quarter_cell
                    end_x = (c[0] + 1) * cell_px_size - quarter_cell
                    start_y = c[1] * cell_px_size + half_cell
                    end_y = (c[1] + 1) * cell_px_size
                elif conn == "west":
                    start_x = c[0] * cell_px_size + quarter_cell
                    end_x = (c[0] + 1) * cell_px_size - quarter_cell
                    start_y = c[1] * cell_px_size
                    end_y = c[1] * cell_px_size + half_cell
                else:
                    start_x = c[0] * cell_px_size + half_cell
                    end_x = (c[0] + 1) * cell_px_size
                    start_y = c[1] * cell_px_size + quarter_cell
                    end_y = (c[1] + 1) * cell_px_size - quarter_cell
                floor_map[start_x: end_x, start_y: end_y, :] = COLORS["GREY"]
        # Draw rooms
        rooms = np.argwhere(self._grid == 1)
        for r in rooms:
            start_x = r[0] * cell_px_size
            end_x = (r[0] + 1) * cell_px_size
            start_y = r[1] * cell_px_size
            end_y = (r[1] + 1) * cell_px_size
            floor_map[start_x: end_x, start_y: end_y, :] = COLORS["BROWN"]
        # Draw agent position
        if agent_pos is not None:
            max_pos = self._grid_size
            #TODO: this works with MiniWorld but doesn't with Godot
            #max_pos = self._grid_size * cell_size
            agent_pos_x = int((agent_pos[0] / max_pos) * self._floor_map_edge) + half_cell
            agent_pos_y = int((agent_pos[1] / max_pos) * self._floor_map_edge) + half_cell
            floor_map[max(agent_pos_y - 1, 0): min(agent_pos_y + 2, self._floor_map_edge), max(agent_pos_x - 1, 0): min(agent_pos_x + 2, self._floor_map_edge), :] = COLORS["RED"]
        #floor_map[agent_pos_y, agent_pos_x, :] = COLORS["RED"]
        # Draw agent direction
        #agent_dir_x = int((agent_dir[0] / max_pos) * self._floor_map_edge)
        #agent_dir_y = int((agent_dir[2] / max_pos) * self._floor_map_edge)
        # TODO: invisible for grid_size >= 5
        #floor_map[max(min(agent_pos_y + agent_dir_y, self._floor_map_edge), 0), max(min(agent_pos_x + agent_dir_x, self._floor_map_edge), 0), :] = COLORS["RED"]
        # Draw goal
        if goal_pos is not None:
            goal_pos_x = int((goal_pos[0] / max_pos) * self._floor_map_edge) + half_cell
            goal_pos_y = int((goal_pos[1] / max_pos) * self._floor_map_edge) + half_cell
            floor_map[goal_pos_y - 1: goal_pos_y + 2, goal_pos_x - 1: goal_pos_x + 2, :] = COLORS["YELLOW"]
        #floor_map[goal_pos_y, goal_pos_x, :] = COLORS["YELLOW"]

        return floor_map
