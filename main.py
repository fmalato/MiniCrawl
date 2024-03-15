import numpy as np
import gymnasium as gym

from minicrawl.controller import BaseController


if __name__ == '__main__':
    controller = BaseController()
    # V1
    #env = gym.make("MiniCrawl-FloorDungeon-v0", render_mode="human")
    # V2
    env = gym.make("MiniCrawl-DungeonMasterEnv-v0", render_mode="human", render_map=True, enlarge_freq=2)
    obs, _ = env.reset(seed=np.random.randint(1, 100000))
    terminated = False
    truncated = False
    while not truncated:
        action = controller.wait_press()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            obs, info = env.env.env.next_level()
