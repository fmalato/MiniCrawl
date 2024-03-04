import numpy as np

from gymnasium import spaces
from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box

from ..params import DEFAULT_PARAMS


class TestDungeon(MiniWorldEnv):
    def __init__(self, max_episode_steps=1500, **kwargs):
        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, params=DEFAULT_PARAMS, **kwargs)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)
        self._room_size = 6

    def _gen_world(self):
        # TODO: Ideas for attributes
        self._grid_width = 2
        self._grid_height = 2
        self._room_size = 6
        # TODO: automatize room creation
        self._rooms = {
            "r1": self.add_rect_room(min_x=6, max_x=12, min_z=6, max_z=12),
            "r2": self.add_rect_room(min_x=6, max_x=12, min_z=-12, max_z=-6),
            "r3": self.add_rect_room(min_x=-12, max_x=-6, min_z=-12, max_z=-6),
            "r4": self.add_rect_room(min_x=-12, max_x=-6, min_z=6, max_z=12)
        }
        self._hallways = {
            "c1": self.add_rect_room(min_x=8, max_x=10, min_z=-6, max_z=6),
            "c2": self.add_rect_room(min_x=-6, max_x=6, min_z=-10, max_z=-8),
            "c3": self.add_rect_room(min_x=-10, max_x=-8, min_z=-6, max_z=6),
            "c4": self.add_rect_room(min_x=-6, max_x=6, min_z=8, max_z=10)
        }
        # TODO: randomize and automatize room edges (graph theory?)
        self.connect_rooms(self._rooms["r1"], self._hallways["c1"], min_x=8, max_x=10)
        self.connect_rooms(self._rooms["r2"], self._hallways["c1"], min_x=8, max_x=10)

        self.connect_rooms(self._rooms["r2"], self._hallways["c2"], min_z=-10, max_z=-8)
        self.connect_rooms(self._rooms["r3"], self._hallways["c2"], min_z=-10, max_z=-8)

        self.connect_rooms(self._rooms["r3"], self._hallways["c3"], min_x=-10, max_x=-8)
        self.connect_rooms(self._rooms["r4"], self._hallways["c3"], min_x=-10, max_x=-8)

        self.connect_rooms(self._rooms["r4"], self._hallways["c4"], min_z=8, max_z=10)
        self.connect_rooms(self._rooms["r1"], self._hallways["c4"], min_z=8, max_z=10)

        # TODO. define goal and scopes
        self.goal = self.place_entity(Box(color="red"))
        self.place_agent()

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        return obs, reward, termination, truncation, info
