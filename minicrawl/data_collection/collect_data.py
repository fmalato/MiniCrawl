import os
import random

import numpy as np
import gymnasium as gym

from minicrawl.controller import BasePynputController


def generate_result_folder_name():
    return ''.join(random.choice('0123456789ABCDEFGHIJLKMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') for i in range(24))


def play_games(controller, num_games, result_dir, render_map):
    for i in range(num_games):
        env = gym.make("MiniCrawl-DungeonCrawlerEnv-v0", render_mode="human", render_map=render_map, boss_stage_freq=4,
                       max_level=20)
        obs, _ = env.reset(seed=np.random.randint(1, 100000))
        # Initialize new run vaiables
        terminated = False
        truncated = False
        total_reward = 0.0
        observations = []
        actions = []
        rewards = []
        new_level = []
        level_names = []
        level_map = []
        has_reached_max = False
        env.render()
        while not truncated:
            action = controller.wait_press()
            obs, reward, terminated, truncated, info = env.step(action)
            # Data gathering
            observations.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            new_level.append(terminated)
            level_map.append(env.unwrapped.get_floor_map_for_analytics())
            # Draw the environment
            env.render()
            # Check for terminal conditions
            if terminated:
                obs, info = env.unwrapped.next_level()
                level_names.append(info["level_name"])
                if env.unwrapped.check_max_level_reached():
                    truncated = True
                    has_reached_max = True
                    total_reward = env.unwrapped.compute_total_reward()
                    env.close()
            if truncated:
                if env.unwrapped.check_max_level_reached():
                    has_reached_max = True
                total_reward = env.unwrapped.compute_total_reward()
                env.close()
        np.savez_compressed(f"{result_dir}/{'map' if render_map else 'no_map'}/game_{i}.npz", observations=observations, actions=actions,
                            level_map=level_map, rewards=rewards, new_level=new_level, has_reached_maximum=has_reached_max,
                            total_reward=total_reward)


def collect_data(num_games):
    original_dir = os.path.abspath(os.curdir)
    result_dir_name = generate_result_folder_name()
    result_dir = os.path.join(original_dir, result_dir_name)

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f"{result_dir}/map", exist_ok=True)
    os.makedirs(f"{result_dir}/no_map", exist_ok=True)

    controller = BasePynputController()
    # Test with map rendering
    play_games(controller, num_games, result_dir, render_map=True)
    # Test without map rendering
    play_games(controller, num_games, result_dir, render_map=False)
