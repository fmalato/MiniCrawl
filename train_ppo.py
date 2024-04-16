import numpy as np
import gymnasium as gym

from stable_baselines3.ppo import PPO

from minicrawl.controller import BasePynputController


if __name__ == '__main__':
    controller = BasePynputController()
    env = gym.make("MiniCrawl-DungeonCrawlerEnv-v0", render_mode="human", render_map=False, boss_stage_freq=4, max_level=20)
    agent = PPO(policy="CnnPolicy", env=env, verbose=2)
    agent.learn(total_timesteps=1000000)
    # Test
    obs, _ = env.reset(seed=np.random.randint(1, 100000))
    terminated = False
    truncated = False
    total_reward = 0.0
    while not truncated:
        action, _ = agent.predict(obs)
        print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    print(f"Total reward: {total_reward}")
