import os
#pyglet.options["headless"] = True

import numpy as np
import minicrawl
import gymnasium as gym
from tqdm import tqdm

import torch

from stable_baselines3.common.policies import ActorCriticPolicy

from resnet_encoder import CausalIDMEncoder
from stable_baselines_bc_agent import TorchObsGymWrapper, decreasing_lr, constant_lr


NUM_GAMES = 100
HISTORY_LENGTH = 4
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    env = gym.make("MiniCrawl-DungeonCrawlerEnv-v0", render_mode="human")
    env = TorchObsGymWrapper(env, history_length=HISTORY_LENGTH)

    net_arch = dict(pi=[256, 256], vf=[256, 256])
    feats_dim = 1024
    lr_schedule = "constant"
    lr_value = constant_lr(0) if lr_schedule == "constant" else decreasing_lr(0)
    use_conv3d = True if HISTORY_LENGTH > 1 else False

    idm_encoder_kwargs = dict(
        feats_dim=feats_dim,
        conv3d_in_channels=HISTORY_LENGTH,
        conv3d_out_channels=128,
        resnet_in_channels=[128 if use_conv3d else 3, 64, 128],
        resnet_out_channels=[64, 128, 128],
        input_size=(1, HISTORY_LENGTH, 3, 60, 80),
        use_conv3d=True,
        device="cuda"
    )
    policy = ActorCriticPolicy(
        observation_space=gym.spaces.Box(low=0.0, high=1.0, shape=(HISTORY_LENGTH, 3, 60, 80)),
        action_space=env.action_space,
        features_extractor_class=CausalIDMEncoder,
        features_extractor_kwargs=idm_encoder_kwargs,
        activation_fn=torch.nn.ReLU,
        net_arch=net_arch,
        lr_schedule=constant_lr if lr_schedule == "constant" else decreasing_lr
    )
    policy.load_state_dict(torch.load("models/bc/best_model_2.pth"))
    policy.action_net.to(DEVICE)

    os.makedirs(f"benchmarks/training/", exist_ok=True)
    num_timesteps = []
    success = []
    ep_durations = []
    ep_rewards = []
    progress_bar = tqdm(range(NUM_GAMES))
    for i in progress_bar:
        observation, info = env.reset()
        truncated = False
        timestep = 0
        ep_reward = 0.0
        while not truncated:
            action, _ = policy.predict(observation, deterministic=False)
            observation, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            timestep += 1
            env.render()

        num_timesteps.append(timestep)
        success.append(1 if ep_reward > 0.0 else 0)
        ep_rewards.append(ep_reward)

        progress_bar.set_description(
            f"[Testing BC] Avg. number of steps: {np.mean(num_timesteps):.3f} +/- {np.std(num_timesteps):.3f} | Success percentage: {np.mean(success) * 100:.3f}% | Avg. reward: {np.mean(ep_rewards):.3f} +/- {np.std(ep_rewards):.3f}")

    avg_steps = np.mean(num_timesteps)
    std_steps = np.std(num_timesteps)
    success_percentage = np.mean(success) * 100
    ep_duration = np.mean(ep_durations)
    ep_duration_std = np.std(ep_durations)
    avg_reward = np.mean(ep_rewards)
    std_reward = np.std(ep_rewards)
    with open(f"benchmarks/training/bc_test.txt", "a+") as f:
        f.write(
            f"Avg. number of steps: {avg_steps:.3f} +/- {std_steps:.3f} | Success percentage: {success_percentage:.3f}% | Avg. episode duration: {ep_duration} +/- {ep_duration_std} | Avg. reward: {avg_reward:.3f} +/- {std_reward:.3f}\n")
    f.close()
