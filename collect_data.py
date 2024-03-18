import os
import random

import numpy as np
import gymnasium as gym

from minicrawl.controller import BaseController


NUM_GAMES = 3

def generate_result_folder_name():
    return ''.join(random.choice('0123456789ABCDEFGHIJLKMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') for i in range(24))


if __name__ == '__main__':
    result_dir = generate_result_folder_name()
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f"{result_dir}/map", exist_ok=True)
    os.makedirs(f"{result_dir}/no_map", exist_ok=True)
    controller = BaseController()
    # Test with map rendering
    env = gym.make("MiniCrawl-DungeonCrawlerEnv-v0", render_mode="human", render_map=True, boss_stage_freq=4,
                   max_level=20)
    obs, _ = env.reset(seed=np.random.randint(1, 100000))
    terminated = False
    truncated = False
    for i in range(NUM_GAMES):
        observations = []
        actions = []
        rewards = []
        new_level = []
        has_reached_max = False
        while not truncated:
            action = controller.wait_press()
            obs, reward, terminated, truncated, info = env.step(action)
            # Data gathering
            observations.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            new_level.append(terminated)
            env.render()
            if terminated:
                obs, info = env.unwrapped.next_level()
                if env.unwrapped.check_max_level_reached():
                    env.close()
                    truncated = True
                    has_reached_max = True
        np.savez_compressed(f"{result_dir}/map/game_{i}.npz", observations=observations, actions=actions,
                            rewards=rewards, new_level=new_level, has_reached_maximum=has_reached_max)
    # Test without map rendering
    env = gym.make("MiniCrawl-DungeonCrawlerEnv-v0", render_mode="human", render_map=False, boss_stage_freq=5,
                   max_level=20)
    obs, _ = env.reset(seed=np.random.randint(1, 100000))
    terminated = False
    truncated = False
    for i in range(NUM_GAMES):
        observations = []
        actions = []
        rewards = []
        new_level = []
        has_reached_max = False
        while not truncated:
            action = controller.wait_press()
            obs, reward, terminated, truncated, info = env.step(action)
            # Data gathering
            observations.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            new_level.append(terminated)
            env.render()
            if terminated:
                obs, info = env.unwrapped.next_level()
                if env.unwrapped.check_max_level_reached():
                    env.close()
                    truncated = True
                    has_reached_max = True
        np.savez_compressed(f"{result_dir}/no_map/game_{i}.npz", observations=observations, actions=actions,
                            rewards=rewards, new_level=new_level, has_reached_maximum=has_reached_max)
