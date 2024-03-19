import math
from abc import abstractmethod
from copy import deepcopy
from typing import Optional, Tuple
from ctypes import POINTER

import gymnasium as gym
import numpy as np
import pyglet.text

from gymnasium.core import ObsType
from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Key, Box, Ball, TextFrame

from minicrawl.dungeon_master import DungeonMaster
from minicrawl.components.rooms import SquaredRoom, JunctionRoom, Corridor
from minicrawl.params import DEFAULT_DM_PARAMS, DEFAULT_PARAMS, BOSS_STAGES, DIRECTIONS

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


class MiniCrawlFloor:
    def __init__(self, max_episode_steps, room_size, floor_tex="wood_planks", wall_tex="cinder_blocks",
                 ceil_tex="rock"):
        self.max_episode_steps = max_episode_steps
        self.room_size = room_size
        self.floor_tex = floor_tex
        self.wall_tex = wall_tex
        self.ceil_tex = ceil_tex

    @abstractmethod
    def step(self, entities, reward, step_count):
        return NotImplementedError

    @abstractmethod
    def gen_world(self, options):
        return NotImplementedError

    @staticmethod
    def near(ent0, ent1):
        """
        Test if the two entities are near each other.
        Used for "go to" or "put next" type tasks

        Taken from miniworld/miniworld.py
        """

        dist = np.linalg.norm(ent0.pos - ent1.pos)
        return dist < ent0.radius + ent1.radius + 1.1 * 0.17

    def _reward(self, step_count):
        return 1.0 - 0.2 * (step_count / self.max_episode_steps)


class MiniCrawlDungeonFloor(MiniCrawlFloor):
    def __init__(self, max_episode_steps=2000, room_size=9, junction_size=3, floor_tex="wood_planks",
                 wall_tex="cinder_blocks",
                 ceil_tex="rock"):
        super().__init__(max_episode_steps, room_size, floor_tex, wall_tex, ceil_tex)
        self.junction_size = junction_size

    def step(self, entities: dict, reward, step_count):
        assert "agent" in entities.keys(), "Stepping in MiniCrawlDungeonFloor requires an agent"
        assert "goal" in entities.keys(), "Stepping in MiniCrawlDungeonFloor requires a goal"
        terminated = False
        truncated = False
        if self.near(entities["agent"], entities["goal"]):
            reward += self._reward(step_count)
            terminated = True
        if step_count > self.max_episode_steps:
            truncated = True

        return reward, terminated, truncated

    def _reward(self, step_count):
        return 1.0 - 0.2 * (step_count / self.max_episode_steps)

    def gen_world(self, options):
        rooms_dict = {}
        junctions_dict = {}
        corrs_dict = {}
        entities_dict = {}
        for i, j in np.ndindex(options["nodes_map"].shape):
            # Build a room
            if options["nodes_map"][i, j] == 1:
                rooms_dict[(i, j)] = SquaredRoom(
                    position=(i, j),
                    edge_size=self.room_size,
                    floor_tex=self.floor_tex,
                    wall_tex=self.wall_tex,
                    ceil_text=self.ceil_tex
                )
            # Build a junction
            if options["nodes_map"][i, j] == 2:
                junctions_dict[(i, j)] = JunctionRoom(
                    position=(i, j),
                    cell_size=self.room_size,
                    floor_tex=self.floor_tex,
                    wall_tex=self.wall_tex,
                    ceil_text=self.ceil_tex
                )
                corrs_dict[(i, j)] = {}
                # Build corridors for junction
                for orientation in options["connections"][(i, j)]:
                    corrs_dict[(i, j)][orientation] = Corridor(
                        position=(i, j),
                        orientation=orientation,
                        cell_size=self.room_size,
                        junction_size=self.junction_size,
                        floor_tex=self.floor_tex,
                        wall_tex=self.wall_tex,
                        ceil_text=self.ceil_tex
                    )

        entities_dict["key"] = Key(color="yellow")

        return rooms_dict, junctions_dict, corrs_dict, entities_dict


class MiniCrawlPutNextFloor(MiniCrawlFloor):
    def __init__(self, max_episode_steps=1000, room_size=12, floor_tex="wood_planks", wall_tex="cinder_blocks",
                 ceil_tex="rock"):
        super().__init__(max_episode_steps, room_size, floor_tex, wall_tex, ceil_tex)
        self.colors = [
            "blue",
            "green",
            "red",
            "yellow"
        ]
        self.positions = [
            [2, 0, 2],
            [self.room_size - 2, 0, 2],
            [2, 0, self.room_size - 2],
            [self.room_size - 2, 0, self.room_size - 2]
        ]

    def step(self, entities, reward, step_count):
        assert "agent" in entities.keys(), "Stepping in PutNextFloor requires an agent"
        assert "target_1" in entities.keys(), "Stepping in PutNextFloor requires target_1"
        assert "target_2" in entities.keys(), "Stepping in PutNextFloor requires target_2"
        terminated = False
        truncated = False
        if entities["agent"].carrying is None and self.near(entities["target_1"], entities["target_2"]):
            reward += self._reward(step_count)
            terminated = True
        if step_count > self.max_episode_steps:
            truncated = True

        return reward, terminated, truncated

    def gen_world(self, options):
        rooms_dict = {}
        entities_dict = {}
        # Selects targets
        targets = deepcopy(self.colors)
        t1 = np.random.choice(targets)
        targets.remove(t1)
        t2 = np.random.choice(targets)
        # Build the room
        rooms_dict[(0, 0)] = SquaredRoom(
            position=(0, 0),
            edge_size=self.room_size,
            floor_tex=self.floor_tex,
            wall_tex=self.wall_tex,
            ceil_text=self.ceil_tex
        )
        # Build cubes and randomly place them in the four corners
        np.random.shuffle(self.colors)
        for color, pos in zip(self.colors, self.positions):
            entities_dict[tuple(pos)] = Box(color=color)
            # Link the same value to two keys for better retrieval later on
            if color == t1:
                entities_dict["target_1"] = entities_dict[tuple(pos)]
            if color == t2:
                entities_dict["target_2"] = entities_dict[tuple(pos)]
        # Add the tags
        upper = [self.room_size, 2.35, self.room_size / 2]
        middle = [self.room_size, 1.70, self.room_size / 2]
        lower = [self.room_size, 1.05, self.room_size / 2]
        entities_dict["upper"] = TextFrame(
            pos=upper,
            dir=math.pi,
            str=t1,
            height=0.65
        )
        entities_dict["middle"] = TextFrame(
            pos=middle,
            dir=math.pi,
            str="near",
            height=0.65
        )
        entities_dict["lower"] = TextFrame(
            pos=lower,
            dir=math.pi,
            str=t2,
            height=0.65
        )

        return rooms_dict, {}, {}, entities_dict


class MiniCrawlPickUpFloor(MiniCrawlFloor):
    def __init__(self, max_episode_steps=1000, room_size=15, floor_tex="wood_planks", wall_tex="cinder_blocks",
                 ceil_tex="rock"):
        super().__init__(max_episode_steps, room_size, floor_tex, wall_tex, ceil_tex)
        self.colors = ["green", "blue", "red", "yellow", "purple", "grey"]
        self.objects = ["box", "key", "ball"]

    def step(self, entities, reward, step_count):
        assert "agent" in entities.keys(), "Stepping in PickUpFloor requires an agent"
        assert "target" in entities.keys(), "Stepping in PickUpFloor requires target"
        terminated = False
        truncated = False
        if entities["agent"].carrying == entities["target"]:
            reward += self._reward(step_count)
            terminated = True
        if entities["agent"].carrying is not None and entities["agent"].carrying != entities["target"]:
            reward = -100
            truncated = True
        if step_count > self.max_episode_steps:
            truncated = True

        return reward, terminated, truncated

    def gen_world(self, options):
        rooms_dict = {}
        entities_dict = {}
        # Build the room
        rooms_dict[(0, 0)] = SquaredRoom(
            position=(0, 0),
            edge_size=self.room_size,
            floor_tex=self.floor_tex,
            wall_tex=self.wall_tex,
            ceil_text=self.ceil_tex
        )
        # Select target
        target_color = np.random.choice(self.colors)
        target_item = np.random.choice(self.objects)
        # Create objects
        pos_x = 1.5
        for c in self.colors:
            pos_z = 1.5
            for o in self.objects:
                if o == "box":
                    entities_dict[(pos_x, 0, pos_z)] = Box(color=c)
                elif o == "key":
                    entities_dict[(pos_x, 0, pos_z)] = Key(color=c)
                else:
                    entities_dict[(pos_x, 0, pos_z)] = Ball(color=c)
                if c == target_color and o == target_item:
                    entities_dict["target"] = entities_dict[(pos_x, 0, pos_z)]
                pos_z += 2.5
            pos_x += 2.5

        # Add mission tags
        upper = [self.room_size, 2.35, self.room_size / 2]
        middle = [self.room_size, 1.70, self.room_size / 2]
        entities_dict["upper"] = TextFrame(
            pos=upper,
            dir=math.pi,
            str="pick up",
            height=0.65
        )
        entities_dict["middle"] = TextFrame(
            pos=middle,
            dir=math.pi,
            str=f"{target_color} {target_item}",
            height=0.65
        )

        return rooms_dict, {},  {}, entities_dict


class MiniCrawlAvoidObstaclesFloor(MiniCrawlFloor):
    def __init__(self, max_episode_steps=1000, room_size=15, floor_tex="wood_planks", wall_tex="cinder_blocks",
                 ceil_tex="rock", num_obstacles=10, obstacles_step=0.1, obstacles_moving=False):
        super().__init__(max_episode_steps, room_size, floor_tex, wall_tex, ceil_tex)
        self.num_obstacles = num_obstacles
        self.obstacles_step = obstacles_step
        self.obstacles_moving = obstacles_moving
        self.obstacles_direction = []

    def step(self, entities, reward, step_count):
        assert "agent" in entities.keys(), "Stepping in PickUpFloor requires an agent"
        assert "target" in entities.keys(), "Stepping in PickUpFloor requires target"
        terminated = False
        truncated = False
        for i, (k, v) in enumerate(entities.items()):
            if isinstance(v, dict):
                direction = v["direction"]
                o = v["entity"]
                if (not 1 < o.pos[0] < self.room_size - 1) or (not 1 < o.pos[2] < self.room_size - 1):
                    v["direction"] = -direction
                if step_count > 0 and self.obstacles_moving:
                    o.pos = o.pos + v["direction"] * self.obstacles_step
                if self.near(entities["agent"], o):
                    reward = -100
                    truncated = True
                    return reward, terminated, truncated

        if self.near(entities["agent"], entities["target"]):
            terminated = True
        if step_count > self.max_episode_steps:
            truncated = True

        return reward, terminated, truncated

    def gen_world(self, options):
        rooms_dict = {}
        entities_dict = {}
        # Build room
        rooms_dict[(0, 0)] = SquaredRoom(
            position=(0, 0),
            edge_size=self.room_size,
            floor_tex=self.floor_tex,
            wall_tex=self.wall_tex,
            ceil_text=self.ceil_tex
        )
        # Build world
        xs = list(range(2, self.room_size - 2, 1))
        zs = list(range(2, self.room_size - 2, 1))
        for i in range(self.num_obstacles):
            self.obstacles_direction.append(np.array(DIRECTIONS[np.random.choice(list(DIRECTIONS.keys()))]))
            pos_x = np.random.choice(xs)
            xs.remove(pos_x)
            pos_z = np.random.choice(zs)
            zs.remove(pos_z)
            entities_dict[(pos_x, 0.5, pos_z)] = {"direction": self.obstacles_direction[i], "entity": Ball(color="blue")}
        # Add target
        entities_dict[(self.room_size - 1, 0, self.room_size - 1)] = Key(color="yellow")
        entities_dict["target"] = entities_dict[(self.room_size - 1, 0, self.room_size - 1)]
        # Add mission
        upper = [self.room_size, 2.35, self.room_size / 2]
        middle = [self.room_size, 1.45, self.room_size / 2]
        entities_dict["upper"] = TextFrame(
            pos=upper,
            dir=math.pi,
            str="avoid blue balls",
            height=0.9
        )
        entities_dict["middle"] = TextFrame(
            pos=middle,
            dir=math.pi,
            str="collect key",
            height=0.9
        )

        return rooms_dict, {}, {}, entities_dict


class MiniCrawlEnv(MiniWorldEnv):
    def __init__(
            self,
            max_episode_steps: int = 2000,
            boss_stage_freq=5,
            dm_kwargs=DEFAULT_DM_PARAMS,
            params=DEFAULT_PARAMS,
            render_map: bool = False,
            max_level: int = 10,
            **kwargs
    ):
        self._dungeon_master = DungeonMaster(**dm_kwargs)
        self.boss_stage_freq = boss_stage_freq
        self.max_level = max_level
        self.rooms_dict = {}
        self.junctions_dict = {}
        self.corrs_dict = {}
        self.entities_dict = {}
        self.step_entities = {}
        self.stairs = None
        self.t1 = None
        self.t2 = None

        self.current_level = 1
        self.step_count = 0
        self.current_floor_name = "dungeon_floor"
        self.current_floor = MiniCrawlDungeonFloor(max_episode_steps=max_episode_steps)

        self.render_map = render_map
        self.params = params
        params.no_random()
        super().__init__(max_episode_steps, params=params, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        reward, terminated, truncated = self.current_floor.step(self.step_entities, reward, self.step_count)

        return obs, reward, terminated, truncated, info

    def reset(
            self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
        self.step_count = 0
        self.rooms_dict = {}
        self.junctions_dict = {}
        self.corrs_dict = {}
        self.entities_dict = {}
        self.step_entities = {}

        if self.current_level % self.boss_stage_freq == 0 and self.current_level != 0:
            boss_stage = np.random.choice(BOSS_STAGES)
            self.current_floor_name = f"{boss_stage}_boss_stage"
        else:
            self.current_floor_name = "dungeon_floor"
        self.current_floor = self._select_new_floor_class()

        obs, info = super().reset(seed=seed, options=options)
        info["level_name"] = self.current_floor_name

        return obs, info

    def next_level(self):
        self.current_level += 1
        self.step_count = 0
        if self.current_level % self.boss_stage_freq == 0 and self.current_level != 0:
            self._dungeon_master.increment_grid_size()
        obs, info = self.reset()

        return obs, info

    def check_max_level_reached(self):
        return self.current_level > self.max_level

    def _select_new_floor_class(self):
        if self.current_level == self.max_level:
            self.current_floor_name = "avoid_obstacles_boss_stage"
            return MiniCrawlAvoidObstaclesFloor(max_episode_steps=1000, obstacles_moving=True)
        if self.current_floor_name == "dungeon_floor":
            return MiniCrawlDungeonFloor(max_episode_steps=self.max_episode_steps)
        elif self.current_floor_name == "put_next_boss_stage":
            return MiniCrawlPutNextFloor(max_episode_steps=500, room_size=12)
        elif self.current_floor_name == "pick_up_boss_stage":
            return MiniCrawlPickUpFloor(max_episode_steps=1000, room_size=15)
        elif self.current_floor_name == "avoid_obstacles_boss_stage":
            return MiniCrawlAvoidObstaclesFloor(max_episode_steps=self.max_episode_steps, obstacles_moving=False)
        else:
            return None

    def _gen_world(self):
        options = {}
        if self.current_floor_name == "dungeon_floor":
            floor_graph, nodes_map, connections = self._dungeon_master.create_dungeon_floor()
            # Dungeon floor requires some options
            options = {
                "floor_grap": floor_graph,
                "nodes_map": nodes_map,
                "connections": connections
            }
        # Build floor
        self.rooms_dict, self.junctions_dict, self.corrs_dict, self.entities_dict = self.current_floor.gen_world(options)
        # Link created entities to MiniWorld entities
        for k, v in self.rooms_dict.items():
            self.rooms.append(v)
        for k, v in self.junctions_dict.items():
            self.rooms.append(v)
            for orient, c in self.corrs_dict[k].items():
                self.rooms.append(c)
        for k, v in self.entities_dict.items():
            if isinstance(k, tuple):
                if isinstance(v, dict):
                    self.entities.append(v["entity"])
                    self.place_entity(v["entity"], pos=np.array(k))
                else:
                    self.entities.append(v)
                    self.place_entity(v, pos=np.array(k))
            # Adjust for TextFrames
            if k in ["upper", "middle", "lower"]:
                self.entities.append(v)
                self.place_entity(v, pos=v.pos, dir=math.pi)
        self._initialize_current_floor()
        self.step_entities["agent"] = self.agent

    def _initialize_current_floor(self):
        # TODO: find a better way to organize this
        if self.current_floor_name == "dungeon_floor":
            self._link_entities()
            goal_room, agent_room = self._dungeon_master.choose_goal_and_agent_positions()
            goal_pos_x, goal_pos_z = self.rooms_dict[goal_room].mid_x, self.rooms_dict[goal_room].mid_z
            stairs = self.place_entity(self.entities_dict["key"], pos=(goal_pos_x, 0, goal_pos_z))
            self.entities.append(stairs)
            self.place_agent(room=self.rooms_dict[agent_room])
            self.step_entities["goal"] = stairs
        elif self.current_floor_name == "put_next_boss_stage":
            self.step_entities["target_1"] = self.entities_dict["target_1"]
            self.step_entities["target_2"] = self.entities_dict["target_2"]
            agent_room = self._dungeon_master.choose_agent_position(self.current_floor_name)
            self.place_agent(room=self.rooms_dict[agent_room], dir=0, min_x=5, max_x=7, min_z=5, max_z=7)
        elif self.current_floor_name == "pick_up_boss_stage":
            self.step_entities["target"] = self.entities_dict["target"]
            agent_room = self._dungeon_master.choose_agent_position(self.current_floor_name)
            self.place_agent(room=self.rooms_dict[agent_room], dir=0, min_x=5, max_x=7, min_z=13, max_z=14)
        elif self.current_floor_name == "avoid_obstacles_boss_stage":
            for k, v in self.entities_dict.items():
                if k != "agent":
                    self.step_entities[k] = v
            agent_room = self._dungeon_master.choose_agent_position(self.current_floor_name)
            self.place_agent(room=self.rooms_dict[agent_room], dir=-math.pi / 4, min_x=0.5, max_x=0.5, min_z=0.5, max_z=0.5)

    def _link_entities(self):
        floor_graph, nodes_map = self._dungeon_master.get_current_floor()
        # Connect corridors with generating junction
        for pos, room in self.junctions_dict.items():
            for orientation in self._dungeon_master.get_connections_for_room(pos):
                corr = self.corrs_dict[pos][orientation]
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
                        room = self.rooms_dict[(i, j)]
                        corr = self.corrs_dict[(i + 1, j)]["north"]
                        self.connect_rooms(room, corr, min_x=corr.min_x, max_x=corr.max_x)
                    # Connect corridor to room
                    elif current_object_type == 2 and object_type == 1:
                        corr = self.corrs_dict[(i, j)][orientation]
                        room = self.rooms_dict[(i + 1, j)]
                        self.connect_rooms(corr, room, min_x=corr.min_x, max_x=corr.max_x)
                    # Connect corridor to corridor
                    elif current_object_type == 2 and object_type == 2:
                        corr1 = self.corrs_dict[(i, j)][orientation]
                        corr2 = self.corrs_dict[(i + 1, j)]["north"]
                        self.connect_rooms(corr1, corr2, min_x=corr1.min_x, max_x=corr1.max_x)
                elif orientation == "east":
                    # If room, neighbors are only corridors
                    if current_object_type == 1:
                        room = self.rooms_dict[(i, j)]
                        corr = self.corrs_dict[(i, j + 1)]["west"]
                        self.connect_rooms(room, corr, min_z=corr.min_z, max_z=corr.max_z)
                    # Connect corridor to room
                    elif current_object_type == 2 and object_type == 1:
                        corr = self.corrs_dict[(i, j)][orientation]
                        room = self.rooms_dict[(i, j + 1)]
                        self.connect_rooms(corr, room, min_z=corr.min_z, max_z=corr.max_z)
                    # Connect corridor to corridor
                    elif current_object_type == 2 and object_type == 2:
                        corr1 = self.corrs_dict[(i, j)][orientation]
                        corr2 = self.corrs_dict[(i, j + 1)]["west"]
                        self.connect_rooms(corr1, corr2, min_z=corr1.min_z, max_z=corr1.max_z)

    def get_floor_map_for_analytics(self):
        return self._dungeon_master.build_floor_map(self.agent.pos, self.agent.dir, self.stairs.pos if self.stairs else np.array([12, 0, 12]))

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
        if self.current_floor_name == "dungeon_floor" and self.render_map:
            floor_map = self._dungeon_master.build_floor_map(self.agent.pos, self.agent.dir_vec, self.step_entities["goal"].pos)
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
        pos_x, pos_y, pos_z = self.agent.pos
        self.text_label.text = "pos: (%.2f, %.2f, %.2f)\nangle: %d\nsteps: %d\nlevel: %d" % (
            pos_x,
            pos_y,
            pos_z,
            int(self.agent.dir * 180 / math.pi) % 360,
            self.step_count,
            self.current_level
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
