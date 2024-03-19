from typing import Optional, Tuple

import numpy as np
from gymnasium.core import ObsType
from miniworld.entity import Key

from minicrawl.minicrawl import MiniCrawlEnv

from minicrawl.components.rooms import SquaredRoom, JunctionRoom, Corridor


class DungeonFloorEnv(MiniCrawlEnv):
    def __init__(
            self,
            floor_graph,
            nodes_map,
            connections,
            stairs_room,
            agent_room,
            render_mode="human",
            **kwargs
    ):
        self.floor_graph = floor_graph
        self.nodes_map = nodes_map
        self.connections = connections
        self.stairs_room = stairs_room
        self.agent_room = agent_room
        super().__init__(render_mode=render_mode, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.near(self.stairs):
            reward += self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
        self.rooms_dict = {}
        self.junctions_dict = {}
        self.corr_dict = {}

        obs, info = super().reset()

        return obs, info
    
    def render(self):
        return super().render()
    
    def _gen_world(self):
        # Build rooms
        for i, j in np.ndindex(self.nodes_map.shape):
            if self.nodes_map[i, j] == 1:
                _ = self.add_room(position=(i, j))
        # Build corridors
        for i, j in np.ndindex(self.nodes_map.shape):
            if self.nodes_map[i, j] == 2:
                room = self.add_junction(position=(i, j))
                # Connect corridors with generating junction
                for orientation in self.connections[(i, j)]:
                    corr = self.add_corridor(position=(i, j), orientation=orientation)
                    if orientation in ["north", "south"]:
                        self.connect_rooms(room, corr, min_x=corr.min_x, max_x=corr.max_x)
                    else:
                        self.connect_rooms(room, corr, min_z=corr.min_z, max_z=corr.max_z)

        # Connect rooms with corridors
        for i, j in np.ndindex(self.nodes_map.shape):
            current_object_type = self.nodes_map[i, j]
            # TODO: reformat code
            for orientation, object_type in self.connections[(i, j)].items():
                if orientation == "south":
                    # If room, neighbors are only corridors
                    if current_object_type == 1:
                        room = self.rooms_dict[i, j]
                        corr = self.corr_dict[i + 1, j]["north"]
                        self.connect_rooms(room, corr, min_x=corr.min_x, max_x=corr.max_x)
                    # Connect corridor to room
                    elif current_object_type == 2 and object_type == 1:
                        corr = self.corr_dict[i, j][orientation]
                        room = self.rooms_dict[i + 1, j]
                        self.connect_rooms(corr, room, min_x=corr.min_x, max_x=corr.max_x)
                    # Connect corridor to corridor
                    elif current_object_type == 2 and object_type == 2:
                        corr1 = self.corr_dict[i, j][orientation]
                        corr2 = self.corr_dict[i + 1, j]["north"]
                        self.connect_rooms(corr1, corr2, min_x=corr1.min_x, max_x=corr1.max_x)
                elif orientation == "east":
                    # If room, neighbors are only corridors
                    if current_object_type == 1:
                        room = self.rooms_dict[i, j]
                        corr = self.corr_dict[i, j + 1]["west"]
                        self.connect_rooms(room, corr, min_z=corr.min_z, max_z=corr.max_z)
                    # Connect corridor to room
                    elif current_object_type == 2 and object_type == 1:
                        corr = self.corr_dict[i, j][orientation]
                        room = self.rooms_dict[i, j + 1]
                        self.connect_rooms(corr, room, min_z=corr.min_z, max_z=corr.max_z)
                    # Connect corridor to corridor
                    elif current_object_type == 2 and object_type == 2:
                        corr1 = self.corr_dict[i, j][orientation]
                        corr2 = self.corr_dict[i, j + 1]["west"]
                        self.connect_rooms(corr1, corr2, min_z=corr1.min_z, max_z=corr1.max_z)

        # Randomly place stairs at the center of one room
        stairs_pos_x, stairs_pos_z = self.rooms_dict[self.stairs_room].mid_x, self.rooms_dict[self.stairs_room].mid_z
        # TODO: provisional. Try to open floor.
        self.stairs = self.place_entity(Key(color="yellow"), pos=(stairs_pos_x, 0, stairs_pos_z))
        """self.stairs = self.place_entity(Stairs(height=1, mesh_name="stairs_down"), pos=(stairs_pos_x, -0.5, stairs_pos_z))
        #self.stairs = self.place_entity(Stairs2D(color="red", tex_name="wood"), pos=(stairs_pos_x, 0, stairs_pos_z))
        # Open a portal in the floor
        floor_portal_verts = {
            "lower_right": self.stairs.pos + np.array([self.stairs.sx, 0.5, -self.stairs.sz]),
            "upper_right": self.stairs.pos + np.array([self.stairs.sx, 0.5, self.stairs.sz]),
            "upper_left": self.stairs.pos + np.array([-self.stairs.sx, 0.5, self.stairs.sz]),
            "lower_left": self.stairs.pos + np.array([-self.stairs.sx, 0.5, -self.stairs.sz])
        }
        self.rooms_dict[rooms_names[stairs_room_idx]].add_portal_on_floor(floor_portal_verts)"""
        starting_room = self.rooms_dict[self.agent_room]
        self.place_agent(room=starting_room)

    def add_room(self, position):
        room = SquaredRoom(position, **self._room_kwargs)
        self.rooms.append(room)
        self.rooms_dict[position] = room

        return room

    def add_junction(self, position):
        junction = JunctionRoom(position, **self._junc_kwargs)
        self.rooms.append(junction)
        self.junctions_dict[position] = junction

        return junction

    def add_corridor(self, position, orientation):
        corr = Corridor(position, orientation, **self._corr_kwargs)
        self.rooms.append(corr)
        if position not in self.corr_dict.keys():
            self.corr_dict[position] = {}
        self.corr_dict[position][orientation] = corr

        return corr
