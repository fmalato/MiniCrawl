import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt

from minicrawl.envs.testdungeon import TestDungeon
from minicrawl.controller import BaseController
from minicrawl.dungeon_master import DungeonMaster


if __name__ == '__main__':
    controller = BaseController()
    env = gym.make("MiniCrawl-FloorDungeon-v0", render_mode="human")
    obs, _ = env.reset(seed=np.random.randint(1, 100000))
    while True:
        action = controller.wait_press()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

