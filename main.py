import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from minicrawl.controller import BasePynputController


if __name__ == '__main__':
    controller = BasePynputController()
    env = gym.make("MiniCrawl-DungeonCrawlerEnv-v0", render_mode="human", render_map=False, boss_stage_freq=3, max_level=3)
    obs, _ = env.reset(seed=np.random.randint(1, 100000))
    terminated = False
    truncated = False
    total_reward = 0.0
    map_images = []
    new_level = []
    while not truncated:
        action = controller.wait_press()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        map_images.append(env.unwrapped.get_floor_map_for_analytics())
        new_level.append(terminated)
        total_reward += reward
    np.savez_compressed("benchmarks/map_plot/no_map_7.npz", level_map=map_images, new_level=new_level)

    print(f"Total reward: {total_reward}")
    env.close()

    env = gym.make("MiniCrawl-DungeonCrawlerEnv-v0", render_mode="human", render_map=True, boss_stage_freq=3, max_level=3)
    obs, _ = env.reset(seed=np.random.randint(1, 100000))
    terminated = False
    truncated = False
    total_reward = 0.0
    map_images = []
    new_level = []
    while not truncated:
        action = controller.wait_press()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        map_images.append(env.unwrapped.get_floor_map_for_analytics())
        new_level.append(terminated)
        total_reward += reward
    np.savez_compressed("benchmarks/map_plot/map_shown_7.npz", level_map=map_images, new_level=new_level)

    print(f"Total reward: {total_reward}")
    env.close()
