from typing import Optional, Tuple

import numpy as np
import pyglet.text
from gymnasium.core import ObsType
from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box

from minicrawl.dungeon_master import DungeonMaster
from minicrawl.components.squared_room import SquaredRoom
from minicrawl.components.corridor import Corridor
from minicrawl.components.junction import JunctionRoom
from minicrawl.params import DEFAULT_PARAMS, DEFAULT_DM_PARAMS, DEFAULT_ROOM_PARAMS, DEFAULT_JUNCTION_PARAMS, DEFAULT_CORRIDOR_PARAMS


class MiniCrawlEnv(MiniWorldEnv):
    def __init__(
            self,
            max_episode_steps: int = 2000,
            obs_width: int = 80,
            obs_height: int = 60,
            window_width: int = 800,
            window_height: int = 600,
            params=DEFAULT_PARAMS,
            domain_rand: bool = False,
            render_mode: Optional[str] = None,
            view: str = "agent",
            dm_kwargs: dict = DEFAULT_DM_PARAMS,
            room_kwargs: dict = DEFAULT_ROOM_PARAMS,
            junc_kwargs: dict = DEFAULT_JUNCTION_PARAMS,
            corr_kwargs: dict = DEFAULT_CORRIDOR_PARAMS

    ):
        self._dungeon_master = DungeonMaster(**dm_kwargs)
        self._level_completed = False
        self._room_kwargs = room_kwargs
        self._junc_kwargs = junc_kwargs
        self._corr_kwargs = corr_kwargs
        self.rooms = []
        super().__init__(max_episode_steps, obs_width, obs_height, window_width, window_height, params, domain_rand,
                         render_mode, view)
        self.rooms_dict = {}
        self.corr_dict = {}
        self.stairs = None
        self.level_label = pyglet.text.Label(
            font_name="Arial",
            font_size=14,
            multiline=True,
            width=400,
            x=window_width + 5,
            y=window_height - (self.obs_disp_height + 19) + 25,
        )

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
        self.corr_dict = {}
        self._dungeon_master.increment_level()

        obs, info = super().reset()

        return obs, info

    def render(self):
        super().render()
        # TODO: add level to text label
        """self.text_label.delete()
        self.text_label.text = self.text_label.text + f"\nLevel: {self._dungeon_master.get_current_level()}"
        self.text_label.draw()"""

    def add_room(self, position):
        room = SquaredRoom(position, **self._room_kwargs)
        self.rooms.append(room)
        self.rooms_dict[position] = room

        return room

    def add_junction(self, position):
        room = JunctionRoom(position, **self._junc_kwargs)
        self.rooms.append(room)
        self.rooms_dict[position] = room

        return room

    def add_corridor(self, position, orientation):
        corr = Corridor(position, orientation, **self._corr_kwargs)
        self.rooms.append(corr)
        if position not in self.corr_dict.keys():
            self.corr_dict[position] = {}
        self.corr_dict[position][orientation] = corr

        return corr

    def _gen_world(self):
        # TODO: sometimes the goal does not spawn inside the floor (connected components?)
        floor_graph, nodes_map = self._dungeon_master.get_current_floor()
        # Build rooms
        for i, j in np.ndindex(nodes_map.shape):
            if nodes_map[i, j] == 1:
                room = self.add_room(position=(i, j))
        # Build corridors
        for i, j in np.ndindex(nodes_map.shape):
            if nodes_map[i, j] == 2:
                room = self.add_junction(position=(i, j))
                # Connect corridors with generating junction
                for orientation in self._dungeon_master.get_connections_for_room((i, j)):
                    corr = self.add_corridor(position=(i, j), orientation=orientation)
                    if orientation in ["north", "south"]:
                        self.connect_rooms(room, corr, min_x=corr.min_x, max_x=corr.max_x)
                    else:
                        self.connect_rooms(room, corr, min_z=corr.min_z, max_z=corr.max_z)

        # Connect rooms with corridors
        for i, j in np.ndindex(nodes_map.shape):
            current_object_type = nodes_map[i, j]
            connections = self._dungeon_master.get_connections_for_room((i, j))
            # TODO: reformat code
            for orientation, object_type in connections.items():
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

        self.stairs = self.place_entity(Box(color="red"))

        self.place_agent()
