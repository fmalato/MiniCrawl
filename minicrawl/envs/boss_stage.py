from typing import Optional, Tuple

import numpy as np
from gymnasium.core import ObsType
from miniworld.entity import Key

from minicrawl.minicrawl import MiniCrawlEnv
from minicrawl.dungeon_master import DungeonMaster
# TODO: maybe reunite in components.py file?
from minicrawl.components.squared_room import SquaredRoom
from minicrawl.components.junction import JunctionRoom
from minicrawl.components.corridor import Corridor
from minicrawl.params import (DEFAULT_ROOM_PARAMS, DEFAULT_JUNCTION_PARAMS, DEFAULT_CORRIDOR_PARAMS,
                              DEFAULT_DM_PARAMS)


class BossStageEnv(MiniCrawlEnv):
    def __init__(self):
        super().__init__()