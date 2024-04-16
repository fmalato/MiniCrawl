import numpy as np
import gymnasium as gym
import wandb

from stable_baselines3.ppo import PPO
from wandb.integration.sb3 import WandbCallback

from minicrawl.controller import BasePynputController


NUM_TIMESTEPS = 10000000
NUM_TEST_GAMES = 30

if __name__ == '__main__':
    config = {
        "env": "MiniCrawl-v0",
        "timesteps": NUM_TIMESTEPS,
        "agent": "PPO",
        "policy": "CnnPolicy"
    }

    run = wandb.init(
        project="minicrawl-benchmarks",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True
    )

    controller = BasePynputController()
    env = gym.make("MiniCrawl-DungeonCrawlerEnv-v0", render_mode="human", render_map=False, boss_stage_freq=4, max_level=20)
    agent = PPO(policy="CnnPolicy", env=env, verbose=1, tensorboard_log=f"runs/{run.id}")
    agent.learn(total_timesteps=NUM_TIMESTEPS, callback=WandbCallback(gradient_save_freq=100, model_save_path=f"models/{run.id}", verbose=2))
    env.close()
    # Test
    env = gym.make("MiniCrawl-DungeonCrawlerEnv-v0", render_mode="human", render_map=False, boss_stage_freq=4,
                   max_level=20)
    for i in range(NUM_TEST_GAMES):
        obs, _ = env.reset(seed=np.random.randint(1, 100000))
        truncated = False
        total_reward = 0.0
        while not truncated:
            action, _ = agent.predict(obs)
            print(action)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

        print(f"Total reward: {total_reward}")

    agent.save("models/ppo.pth")
