import math
from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
from gymnasium.core import ObsType
from miniworld.entity import Box, TextFrame

from minicrawl.minicrawl import MiniCrawlEnv
from minicrawl.params import DEFAULT_ROOM_PARAMS


class BossStageEnv(MiniCrawlEnv):
    def __init__(self, room_size, max_episode_steps=2000, render_mode="human", view="agent",  **kwargs):
        super().__init__(render_mode=render_mode, max_episode_steps=max_episode_steps, view=view,  **kwargs)
        self.room_size = room_size

    @abstractmethod
    def step(self, action):
        return NotImplementedError

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)

    @abstractmethod
    def _gen_world(self):
        return NotImplementedError

    @abstractmethod
    def compute_reward(self):
        return NotImplementedError


class PutNextBossStageEnv(BossStageEnv):
    def __init__(self, room_size=12, render_mode="human", view="agent", max_episode_steps=2000,  **kwargs):
        self.max_episode_steps = max_episode_steps
        super().__init__(room_size=room_size, render_mode=render_mode, view=view,  **kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        t1 = self.putnext_cubes[self.putnext_targets["t1"]]
        t2 = self.putnext_cubes[self.putnext_targets["t2"]]
        if (not self.agent.carrying) and self.near(t1, t2):
            reward += self._reward()
            terminated = True
            
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
        return super().reset(seed=seed, options=options)
    
    def render(self):
        return super().render()

    def _gen_world(self):
        room = self.add_room(pos=(0, 0))
        blue = self.place_entity(Box(color="blue"), pos=np.array([2, 0, 2]))
        red = self.place_entity(Box(color="red"), pos=np.array([2, 0, self.room_size - 2]))
        green = self.place_entity(Box(color="green"), pos=np.array([self.room_size - 2, 0, 2]))
        yellow = self.place_entity(Box(color="yellow"), pos=np.array([self.room_size - 2, 0, self.room_size - 2]))
        self.putnext_cubes = {
            "blue": blue,
            "red": red,
            "green": green,
            "yellow": yellow
        }
        self.place_agent(room, dir=-math.pi, min_x=4, max_x=5, min_z=4, max_z=5)
        targets = list(self.putnext_cubes.keys())
        t1 = np.random.choice(targets)
        targets.remove(t1)
        t2 = np.random.choice(targets)
        self.putnext_targets = {
            "t1": t1,
            "t2": t2
        }

        t1_sign = TextFrame(
            pos=[DEFAULT_ROOM_PARAMS["edge_size"], 2.35, DEFAULT_ROOM_PARAMS["edge_size"] / 2],
            dir=math.pi,
            str=t1,
            height=0.65
        )
        middle_sign = TextFrame(
            pos=[DEFAULT_ROOM_PARAMS["edge_size"], 1.70, DEFAULT_ROOM_PARAMS["edge_size"] / 2],
            dir=math.pi,
            str="near",
            height=0.65
        )
        t2_sign = TextFrame(
            pos=[DEFAULT_ROOM_PARAMS["edge_size"], 1.05, DEFAULT_ROOM_PARAMS["edge_size"] / 2],
            dir=math.pi,
            str=t2,
            height=0.6
        )
        self.entities.append(t1_sign)
        self.entities.append(middle_sign)
        self.entities.append(t2_sign)
