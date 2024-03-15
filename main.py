import numpy as np
import gymnasium as gym

from minicrawl.controller import BaseController


if __name__ == '__main__':
    controller = BaseController()
    env = gym.make("MiniCrawl-DungeonCrawlerEnv-v0", render_mode="human", render_map=True, boss_stage_freq=2, max_level=4)
    obs, _ = env.reset(seed=np.random.randint(1, 100000))
    terminated = False
    truncated = False
    obs = []
    acts = []

    while not truncated:
        action = controller.wait_press()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            obs, info = env.unwrapped.next_level()
            if env.unwrapped.check_max_level_reached():
                env.close()
                truncated = True
