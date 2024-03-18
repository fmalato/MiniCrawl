from miniworld.params import DomainParams
from enum import Enum

import numpy as np


# Default simulation parameters
DEFAULT_PARAMS = DomainParams()
DEFAULT_PARAMS.set("sky_color", [0.25, 0.82, 1], [0.1, 0.1, 0.1], [1.0, 1.0, 1.0])
DEFAULT_PARAMS.set("light_pos", [0, 2.5, 0], [-40, 2.5, -40], [40, 5, 40])
DEFAULT_PARAMS.set("light_color", [0.7, 0.7, 0.7], [0.45, 0.45, 0.45], [0.8, 0.8, 0.8])
DEFAULT_PARAMS.set(
    "light_ambient", [0.45, 0.45, 0.45], [0.35, 0.35, 0.35], [0.55, 0.55, 0.55]
)
DEFAULT_PARAMS.set("obj_color_bias", [0, 0, 0], [-0.2, -0.2, -0.2], [0.2, 0.2, 0.2])
DEFAULT_PARAMS.set("forward_step", 0.15, 0.15, 0.15)
DEFAULT_PARAMS.set("forward_drift", 0, -0.05, 0.05)
DEFAULT_PARAMS.set("turn_step", 3, 3, 3)
DEFAULT_PARAMS.set("bot_radius", 0.4, 0.38, 0.42)
DEFAULT_PARAMS.set("cam_pitch", 0, -5, 5)
DEFAULT_PARAMS.set("cam_fov_y", 60, 55, 65)
DEFAULT_PARAMS.set("cam_height", 1.5, 1.45, 1.55)
DEFAULT_PARAMS.set("cam_fwd_disp", 0, -0.05, 0.10)


DEFAULT_DM_PARAMS = dict(
    starting_grid_size=3,
    max_grid_size=None,
    increment_freq=2,
    connection_density=0.75
)

DEFAULT_CELL_SIZE = 9
DEFAULT_EDGE_SIZE = 3

DEFAULT_ROOM_PARAMS = dict(
    edge_size=DEFAULT_CELL_SIZE,
    wall_height=2.74,
    floor_tex="wood_planks",
    wall_tex="cinder_blocks",
    ceil_text="rock",
    no_ceiling=False
)

DEFAULT_JUNCTION_PARAMS = dict(
    cell_size=DEFAULT_CELL_SIZE,
    edge_size=DEFAULT_EDGE_SIZE,
    wall_height=2.74,
    floor_tex="wood_planks",
    wall_tex="cinder_blocks",
    ceil_text="rock",
    no_ceiling=False
)

DEFAULT_CORRIDOR_PARAMS = dict(
    cell_size=DEFAULT_CELL_SIZE,
    junction_size=DEFAULT_EDGE_SIZE,
    wall_height=2.74,
    floor_tex="wood_planks",
    wall_tex="cinder_blocks",
    ceil_text="rock",
    no_ceiling=False
)

COLORS = dict(
    BROWN=np.array([123, 63, 0], dtype=np.uint8),
    GREY=np.array([128, 128, 128], dtype=np.uint8),
    RED=np.array([255, 0, 0], dtype=np.uint8),
    YELLOW=np.array([255, 255, 0], dtype=np.uint8)
)

# TODO: design more
BOSS_STAGES = [
    "avoid_obstacles"
]

DIRECTIONS = {
    "north": [0, 0, -1],
    "south": [0, 0, 1],
    "east": [1, 0, 0],
    "west": [-1, 0, 0]
}
