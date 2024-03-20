import numpy as np
import gymnasium as gym

from minicrawl.controller import BaseController


if __name__ == '__main__':
    controller = BaseController()
    env = gym.make("MiniCrawl-DungeonCrawlerEnv-v0", render_mode="human", render_map=True, boss_stage_freq=2, max_level=6)
    obs, _ = env.reset(seed=np.random.randint(1, 100000))
    terminated = False
    truncated = False
    total_reward = 0.0
    while not truncated:
        action = controller.wait_press()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            obs, info = env.unwrapped.next_level()
            print(f"Current level reward: {reward}")
            if env.unwrapped.check_max_level_reached():
                env.close()
                truncated = True
                total_reward = env.unwrapped.compute_total_reward()
        if truncated:
            print(f"Current level reward: {reward}")
            total_reward = env.unwrapped.compute_total_reward()
            env.close()

    print(f"Total reward: {total_reward}")
