import gymnasium as gym
import numpy as np
import torch
import minicrawl

from stable_baselines3.ppo import PPO
from tqdm import tqdm


NUM_TEST_GAMES = 100
RENDER = True
MODEL_PATH = "models/ppo/ppo.pth"


if __name__ == '__main__':
    # Nice print
    np.set_printoptions(precision=3)

    env = gym.make("MiniCrawl-DungeonCrawlerEnv-v0", render_mode="human", render_map=False, boss_stage_freq=4,
                   max_level=20)
    agent = PPO(policy="CnnPolicy", env=env, verbose=1)
    agent.load(MODEL_PATH)

    progress_bar = tqdm(range(NUM_TEST_GAMES))

    for i in progress_bar:
        obs, _ = env.reset(seed=np.random.randint(1, 100000))
        truncated = False
        total_reward = 0.0
        action_distrib = np.zeros(shape=env.action_space.n)
        timestep = 0
        while not truncated:
            action, _ = agent.predict(obs, deterministic=False)
            action_distrib[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            timestep += 1
            if RENDER:
                progress_bar.set_description(f"Action distribution: {action_distrib / timestep}")
                env.render()

        print(f"Total reward: {total_reward}")
