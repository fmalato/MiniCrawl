import math
from abc import abstractmethod
from typing import Optional, Tuple
from ctypes import POINTER

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pyglet.text
from gymnasium.core import ObsType
from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Key, Box, TextFrame

from minicrawl.dungeon_master import DungeonMaster
from minicrawl.components.squared_room import SquaredRoom
from minicrawl.components.corridor import Corridor
from minicrawl.components.junction import JunctionRoom
from minicrawl.components.geometric_entities import Stairs, Stairs2D
from minicrawl.params import (DEFAULT_PARAMS, DEFAULT_DM_PARAMS, DEFAULT_ROOM_PARAMS, DEFAULT_JUNCTION_PARAMS,
                              DEFAULT_CORRIDOR_PARAMS, ENV_NAMES)

from pyglet.gl import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_FRAMEBUFFER,
    GL_MODELVIEW,
    GL_PROJECTION,
    GLubyte,
    glBindFramebuffer,
    glClear,
    glClearColor,
    glClearDepth,
    glFlush,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
)


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
            render_map: bool = False,
            view: str = "agent",
            dm_kwargs: dict = DEFAULT_DM_PARAMS,
            room_kwargs: dict = DEFAULT_ROOM_PARAMS,
            junc_kwargs: dict = DEFAULT_JUNCTION_PARAMS,
            corr_kwargs: dict = DEFAULT_CORRIDOR_PARAMS

    ):
        self._dungeon_master = DungeonMaster(**dm_kwargs)
        self._render_map = render_map
        self._level_type = "dungeon_floor"
        self._room_kwargs = room_kwargs
        self._junc_kwargs = junc_kwargs
        self._corr_kwargs = corr_kwargs
        self.rooms = []
        params.no_random()
        super().__init__(max_episode_steps, obs_width, obs_height, window_width, window_height, params, domain_rand,
                         render_mode, view)
        self.rooms_dict = {}
        self.junctions_dict = {}
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
        # TODO: This won't scale well. Re-design levels hierarchy.
        if self._level_type == "dungeon_floor" and self.near(self.stairs):
            reward += self._reward()
            terminated = True
        elif self._level_type == "put_next_boss_stage":
            t1 = self.putnext_cubes[self.putnext_targets["t1"]]
            t2 = self.putnext_cubes[self.putnext_targets["t2"]]
            if (not self.agent.carrying) and self.near(t1, t2):
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
        """
        Renders the environment for humans.
        Exact same as miniworld, but label is updated to include also the level.
        """

        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        # Render the human-view image
        if self.view == "agent":
            img = self.render_obs(self.vis_fb)
        else:
            img = self.render_top_view(self.vis_fb)
        img_width = img.shape[1]
        img_height = img.shape[0]

        if self.render_mode == "rgb_array":
            return img

        # Render the agent's view
        obs = self.render_obs()
        obs_width = obs.shape[1]
        obs_height = obs.shape[0]

        window_width = img_width + self.obs_disp_width
        window_height = img_height

        if self.window is None:
            config = pyglet.gl.Config(double_buffer=True)
            self.window = pyglet.window.Window(
                width=window_width, height=window_height, resizable=False, config=config
            )

        self.window.clear()
        self.window.switch_to()

        # Bind the default frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Clear the color and depth buffers
        glClearColor(0, 0, 0, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Setup orghogonal projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glOrtho(0, window_width, 0, window_height, 0, 10)

        # Draw the human render to the rendering window
        img_flip = np.ascontiguousarray(np.flip(img, axis=0))
        img_data = pyglet.image.ImageData(
            img_width,
            img_height,
            "RGB",
            img_flip.ctypes.data_as(POINTER(GLubyte)),
            pitch=img_width * 3,
        )
        img_data.blit(0, 0, 0, width=img_width, height=img_height)

        # Draw the observation
        obs = np.ascontiguousarray(np.flip(obs, axis=0))
        obs_data = pyglet.image.ImageData(
            obs_width,
            obs_height,
            "RGB",
            obs.ctypes.data_as(POINTER(GLubyte)),
            pitch=obs_width * 3,
        )
        obs_data.blit(
            img_width,
            img_height - self.obs_disp_height,
            0,
            width=self.obs_disp_width,
            height=self.obs_disp_height,
        )

        # Draw map
        if self._render_map:
            floor_map = self._dungeon_master.build_floor_map(self.agent.pos, self.agent.dir_vec, self.stairs.pos)
            floor_map = np.ascontiguousarray(np.flip(floor_map, axis=0))
            map_data = pyglet.image.ImageData(
                obs_width,
                obs_width,
                "RGB",
                floor_map.ctypes.data_as(POINTER(GLubyte)),
                pitch=obs_width * 3,
            )
            map_data.blit(
                img_width,
                img_height - self.obs_disp_height - 350,
                0,
                width=self.obs_disp_width,
                height=self.obs_disp_width,
            )

        # Draw the text label in the window
        self.text_label.text = "pos: (%.2f, %.2f, %.2f)\nangle: %d\nsteps: %d\nlevel: %d" % (
            *self.agent.pos,
            int(self.agent.dir * 180 / math.pi) % 360,
            self.step_count,
            self._dungeon_master.get_current_level()
        )
        self.text_label.draw()

        # Force execution of queued commands
        glFlush()

        # If we are not running the Pyglet event loop,
        # we have to manually flip the buffers and dispatch events
        if self.render_mode == "human":
            self.window.flip()
            self.window.dispatch_events()

            return

        return img

    def next_level(self):
        self._level_type = self._dungeon_master.increment_level()
        obs, info = self.reset()

        return obs, info

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

    def _gen_world(self):
        if self._level_type == "dungeon_floor":
            self._gen_dungeon_floor()
        else:
            if self._level_type.startswith("put_next"):
                self._gen_putnext_stage()
            elif self._level_type.startswith("follow_instructions"):
                self._gen_follow_stage()

    def _gen_dungeon_floor(self):
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

        # Randomly place stairs at the center of one room
        stairs_room, agent_room = self._dungeon_master.choose_goal_and_agent_positions()
        stairs_pos_x, stairs_pos_z = self.rooms_dict[stairs_room].mid_x, self.rooms_dict[stairs_room].mid_z
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
        starting_room = self.rooms_dict[agent_room]
        self.place_agent(room=starting_room)

    def _gen_putnext_stage(self):
        room = self.add_room((0, 0))
        blue = self.place_entity(Box(color="blue"), pos=np.array([1, 0, 1]))
        red = self.place_entity(Box(color="red"), pos=np.array([1, 0, self._room_kwargs["edge_size"] - 1]))
        green = self.place_entity(Box(color="green"), pos=np.array([self._room_kwargs["edge_size"] - 1, 0, 1]))
        yellow = self.place_entity(Box(color="yellow"), pos=np.array([self._room_kwargs["edge_size"] - 1, 0, self._room_kwargs["edge_size"] - 1]))
        self.putnext_cubes = {
            "blue": blue,
            "red": red,
            "green": green,
            "yellow": yellow
        }
        self.place_agent(room, dir=math.pi, min_x=4, max_x=5, min_z=4, max_z=5)
        targets = list(self.putnext_cubes.keys())
        t1 = np.random.choice(targets)
        targets.remove(t1)
        t2 = np.random.choice(targets)
        self.putnext_targets = {
            "t1": t1,
            "t2": t2
        }

        t1_sign = TextFrame(
            pos=[self._room_kwargs["edge_size"], 2.35, self._room_kwargs["edge_size"] / 2],
            dir=math.pi,
            str=t1,
            height=0.65
        )
        middle_sign = TextFrame(
            pos=[self._room_kwargs["edge_size"], 1.70, self._room_kwargs["edge_size"] / 2],
            dir=math.pi,
            str="near",
            height=0.65
        )
        t2_sign = TextFrame(
            pos=[self._room_kwargs["edge_size"], 1.05, self._room_kwargs["edge_size"] / 2],
            dir=math.pi,
            str=t2,
            height=0.6
        )
        self.entities.append(t1_sign)
        self.entities.append(middle_sign)
        self.entities.append(t2_sign)


class MiniCrawlEnvV2(MiniWorldEnv):
    def __init__(
            self,
            max_episode_steps: int = 2000,
            **kwargs
    ):
        super().__init__(max_episode_steps, **kwargs)

    @abstractmethod
    def step(self, action):
        return NotImplementedError()

    @abstractmethod
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
        return NotImplementedError()

    def render(self):
        """
        Renders the environment for humans.
        Exact same as miniworld, but label is updated to include also the level.
        """

        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        # Render the human-view image
        if self.view == "agent":
            img = self.render_obs(self.vis_fb)
        else:
            img = self.render_top_view(self.vis_fb)
        img_width = img.shape[1]
        img_height = img.shape[0]

        if self.render_mode == "rgb_array":
            return img

        # Render the agent's view
        obs = self.render_obs()
        obs_width = obs.shape[1]
        obs_height = obs.shape[0]

        window_width = img_width + self.obs_disp_width
        window_height = img_height

        if self.window is None:
            config = pyglet.gl.Config(double_buffer=True)
            self.window = pyglet.window.Window(
                width=window_width, height=window_height, resizable=False, config=config
            )

        self.window.clear()
        self.window.switch_to()

        # Bind the default frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Clear the color and depth buffers
        glClearColor(0, 0, 0, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Setup orthogonal projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glOrtho(0, window_width, 0, window_height, 0, 10)

        # Draw the human render to the rendering window
        img_flip = np.ascontiguousarray(np.flip(img, axis=0))
        img_data = pyglet.image.ImageData(
            img_width,
            img_height,
            "RGB",
            img_flip.ctypes.data_as(POINTER(GLubyte)),
            pitch=img_width * 3,
        )
        img_data.blit(0, 0, 0, width=img_width, height=img_height)

        # Draw the observation
        obs = np.ascontiguousarray(np.flip(obs, axis=0))
        obs_data = pyglet.image.ImageData(
            obs_width,
            obs_height,
            "RGB",
            obs.ctypes.data_as(POINTER(GLubyte)),
            pitch=obs_width * 3,
        )
        obs_data.blit(
            img_width,
            img_height - self.obs_disp_height,
            0,
            width=self.obs_disp_width,
            height=self.obs_disp_height,
        )

        # Draw map
        """if self._render_map:
            floor_map = self._dungeon_master.build_floor_map(self.agent.pos, self.agent.dir_vec, self.stairs.pos)
            floor_map = np.ascontiguousarray(np.flip(floor_map, axis=0))
            map_data = pyglet.image.ImageData(
                obs_width,
                obs_width,
                "RGB",
                floor_map.ctypes.data_as(POINTER(GLubyte)),
                pitch=obs_width * 3,
            )
            map_data.blit(
                img_width,
                img_height - self.obs_disp_height - 350,
                0,
                width=self.obs_disp_width,
                height=self.obs_disp_width,
            )"""

        # Draw the text label in the window
        self.text_label.text = "pos: (%.2f, %.2f, %.2f)\nangle: %d\nsteps: %d\nlevel: %d" % (
            *self.agent.pos,
            int(self.agent.dir * 180 / math.pi) % 360,
            self.step_count,
            self._dungeon_master.get_current_level()
        )
        self.text_label.draw()

        # Force execution of queued commands
        glFlush()

        # If we are not running the Pyglet event loop,
        # we have to manually flip the buffers and dispatch events
        if self.render_mode == "human":
            self.window.flip()
            self.window.dispatch_events()

            return

        return img

    def _gen_world(self):
        env_name = ENV_NAMES[self._level_type]
        self._current_level = gym.make(env_name, max_episode_steps=2000)
        self._current_level.reset()