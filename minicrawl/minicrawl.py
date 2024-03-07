from typing import Optional, Tuple

import numpy as np
from gymnasium.core import ObsType
from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box
from miniworld.params import DEFAULT_PARAMS

from minicrawl.dungeon_master import DungeonMaster
from minicrawl.components.squared_room import SquaredRoom
from minicrawl.components.corridor import Corridor
from minicrawl.components.junction import JunctionRoom
from minicrawl.params import DEFAULT_DM_PARAMS, DEFAULT_ROOM_PARAMS, DEFAULT_JUNCTION_PARAMS, DEFAULT_CORRIDOR_PARAMS


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

    def step(self, action):
        return super().step(action)

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
        nodes = floor_graph.get_nodes()
        for n in nodes:
            if nodes_map[n[0], n[1]] == 1:
                room = self.add_room(position=n)
            elif nodes_map[n[0], n[1]] == 2:
                room = self.add_junction(position=n)
                for orientation in self._dungeon_master.get_connections_for_room(n):
                    corr = self.add_corridor(position=n, orientation=orientation)
                    if orientation in ["north", "south"]:
                        self.connect_rooms(room, corr, min_x=corr.min_x, max_x=corr.max_x)
                    else:
                        self.connect_rooms(room, corr, min_z=corr.min_z, max_z=corr.max_z)

        # Connect rooms with corridors
        for pos, corrs in self.corr_dict.items():
            for orientation, corr in corrs.items():
                if orientation == "north":
                    self.connect_rooms(self.rooms_dict[pos[0] - 1, pos[1]], corr, min_x=corr.min_x, max_x=corr.max_x)
                elif orientation == "south":
                    self.connect_rooms(self.rooms_dict[pos[0] + 1, pos[1]], corr, min_x=corr.min_x, max_x=corr.max_x)
                elif orientation == "east":
                    self.connect_rooms(self.rooms_dict[pos[0], pos[1] + 1], corr, min_z=corr.min_z, max_z=corr.max_z)
                else:
                    self.connect_rooms(self.rooms_dict[pos[0], pos[1] - 1], corr, min_z=corr.min_z, max_z=corr.max_z)
        """for n in nodes:
            if nodes_map[n[0], n[1]] == 1:
                for orientation, c in self.corr_dict[n].items():
                    self.connect_rooms(self.rooms_dict[n], c, c.min_x, c.max_x)"""

        self.box = self.place_entity(Box(color="red"))

        self.place_agent()
