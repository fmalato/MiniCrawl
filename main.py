import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt

from minicrawl.controller import BaseController


if __name__ == '__main__':
    controller = BaseController()
    env = gym.make("MiniCrawl-FloorDungeon-v0", render_mode="human")
    obs, _ = env.reset(seed=np.random.randint(1, 100000))
    get_map = True
    terminated = False
    while not terminated:
        action = controller.wait_press()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if get_map:
            map_array = env.render_top_view()
            plt.imsave("floor_map.png", map_array)
            get_map = False
