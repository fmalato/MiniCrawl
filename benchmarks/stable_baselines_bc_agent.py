import os
import argparse
from typing import SupportsFloat, Any

import numpy as np
import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType
from tqdm import tqdm

import wandb
import torch

from torch.utils.data import Dataset, DataLoader
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium.spaces.box as box

from imitation.algorithms import bc
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env

from resnet_encoder import CausalIDMEncoder


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class TorchObsGymWrapper(gym.Wrapper):
    def __init__(self, env, history_length=1):
        super().__init__(env=env)
        obs_shape = env.observation_space.shape
        self.observation_space = box.Box(low=0, high=1, shape=(history_length, obs_shape[2], obs_shape[0], obs_shape[1]), dtype=np.float32)
        self.history_length = history_length
        self.obs_buffer = [np.zeros(shape=self.observation_space.shape[1:]) for i in range(history_length)]

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        self.obs_buffer.pop(0)
        self.obs_buffer.append(obs.transpose((2, 0, 1)).astype(np.float32) / 255)

        return self.obs_buffer, reward, terminated, truncated, info

    def reset(
        self, *, seed: int = None, options: dict[str, Any] = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = super().reset()
        self.obs_buffer = [np.zeros(shape=self.observation_space.shape[1:]) for i in range(self.history_length)]
        self.obs_buffer.pop(0)
        self.obs_buffer.append(obs.transpose((2, 0, 1)).astype(np.float32) / 255)

        return self.obs_buffer, info

    def seed(self, s):
        super().seed(s)


def constant_lr(timestep: float) -> float:
    return 1e-4


def decreasing_lr(timestep: float) -> float:
    start = 3e-4
    end = 5e-5
    print((start - end / 307000) * timestep + start)

    return (start - end / 307000) * timestep + start


"""class FreeplayObs(ObsBuilder):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        ball = state.ball

        obs = [
            ball.position,
            ball.linear_velocity,
            ball.angular_velocity
        ]

        _ = self._add_player_to_obs(obs, player)

        return np.concatenate(obs)

    def _add_player_to_obs(self, obs, player):
        player_car = player.car_data

        obs.extend([
            player_car.position,
            player_car.linear_velocity,
            player_car.angular_velocity
        ])

        return player_car"""


class MiniWorldDataset(Dataset):
    def __init__(self, data_path, history_length=4, use_all=False):
        super().__init__()
        self.data_path = data_path
        self.history_length = history_length
        self.traj_metadata = np.genfromtxt(os.path.join(data_path, "trajectories_lengths.csv"), delimiter=",")
        # Remove fails if requested (valid only for examples where success/failure can be determined)
        if not use_all:
            failures = np.where(self.traj_metadata[:, 1] == 1000)
            self.traj_metadata = np.delete(self.traj_metadata, failures, axis=0)
        # Compute offsets
        self._compute_offsets()
        # Avoids repetitive computation later on
        self._num_samples = int(np.sum(self.traj_metadata))

    def __len__(self):
        return self._num_samples

    def __getitem__(self, item):
        # Find belonging bin
        start, end = self.traj_metadata[:, 2], self.traj_metadata[:, 3]
        mask = (item >= start) & (item <= end)
        traj_idx = np.argmax(mask)
        # Load data
        traj = np.load(os.path.join(self.data_path, f"{int(self.traj_metadata[traj_idx, 0])}.npz"))
        # Build observation
        offset = item - int(start[traj_idx])
        if offset < self.history_length - 1:
            zeros_stack = np.zeros(shape=(self.history_length - 1 - offset, *traj["observations"][0, :].shape))
            obs = traj["observations"][0: offset + 1, :]
            obs = np.concatenate([zeros_stack, obs], axis=0)
        else:
            obs = traj["observations"][offset - self.history_length + 1: offset + 1, :]
        act = traj["actions"][item - int(start[traj_idx])]

        return {"obs": torch.FloatTensor(obs.astype(np.float32) / 255).permute((0, 3, 1, 2)), "act": act}

    def _compute_offsets(self):
        total_count = 0
        offsets = np.zeros(shape=self.traj_metadata.shape)
        for i, v in enumerate(list(self.traj_metadata[:, 1])):
            offsets[i, 0] = total_count
            offsets[i, 1] = total_count + v - 1
            total_count += v

        self.traj_metadata = np.append(self.traj_metadata, offsets, axis=1)


def train_bc(agent, data_loader, epochs, device="cuda"):
    assert device in ["cuda", "cpu"], "Unknown device."
    progress_bar = tqdm(enumerate(data_loader))
    for e in range(epochs):
        progress_bar.set_description(f"Epoch {e}")
        for i, batch in progress_bar:
            agent.optimizer.zero_grad()
            # Forward pass + metrics computation
            training_metrics = agent.loss_calculator(agent.policy, batch["obs"].to(device), batch["act"].to(device))
            # Compute loss (rescale if minibatch_size != batch_size, see original BC code)
            loss = training_metrics.loss * batch["act"].shape[0] / agent.batch_size
            loss.backward()
            agent.optimizer.step()
            progress_bar.set_description(f"Epoch {e} - Loss: {loss.item()}")

    return agent

def collate_fn(batch):
    return {
        'observations': batch["obs"],
        'actions': torch.tensor([x['labels'] for x in batch])
    }


def main(args):
    ds = MiniWorldDataset("gail_recorded/", history_length=args.history_length)
    data_loader = DataLoader(ds, batch_size=args.batch_size, num_workers=8)

    net_arch = dict(pi=[256, 256] if args.env_name != "rlgym" else [64, 64], vf=[256, 256] if args.env_name != "rlgym" else [64, 64])
    feats_dim = 512 if args.env_name != "rlgym" else 64
    lr_schedule = "constant"
    lr_value = constant_lr(0) if lr_schedule == "constant" else decreasing_lr(0)

    if args.use_wandb:
        wandb.init(
            project="BC+ZIP Adaptation",
            notes=f"BC baseline SB3 training - Env: {args.env_name}",
            tags=["bc", "baseline", "stable-baselines3", args.env_name],
            config={
                "epochs": args.epochs,
                "net_arch": net_arch,
                "features dimension": feats_dim,
                "lr_schedule": lr_schedule,
                "lr_value": lr_value,
                "trajectories": args.trajectories_path,
                "num_trajectories": len(os.listdir(args.trajectories_path))
            },
            sync_tensorboard=True
        )

    rng = np.random.default_rng(0)
    env = make_vec_env(
        env_name=args.env_name,
        rng=rng,
        n_envs=1,
        post_wrappers=[lambda env, _: TorchObsGymWrapper(RolloutInfoWrapper(env))],
        env_make_kwargs=dict(render_mode="human", render_map=False, boss_stage_freq=4, max_level=20)
    )

    idm_encoder_kwargs = dict(
        feats_dim=1024,
        conv3d_in_channels=args.history_length,
        conv3d_out_channels=128,
        resnet_in_channels=[128, 64, 128],
        resnet_out_channels=[64, 128, 128],
        input_size=(1, args.history_length, 3, 60, 80),
        use_conv3d=True,
        device="cuda"
    )
    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        features_extractor_class=CausalIDMEncoder,
        features_extractor_kwargs=idm_encoder_kwargs,
        activation_fn=torch.nn.ReLU,
        net_arch=net_arch,
        lr_schedule=constant_lr if lr_schedule == "constant" else decreasing_lr
    )

    bc_trainer = bc.BC(
        policy=policy,
        observation_space=env.observation_space if args.env_name != "rlgym" else gym.spaces.box.Box(low=0.0, high=1.0, shape=(18,)),
        action_space=env.action_space if args.env_name != "rlgym" else gym.spaces.box.Box(low=-1.0, high=1.0, shape=(8,)),
        demonstrations=None,
        rng=rng
    )
    os.makedirs(f"models/bc/{args.env_name}", exist_ok=True)
    os.makedirs(f"logs/{args.env_name}", exist_ok=True)
    with open(f"logs/{args.env_name}/test_results_bc.txt", "a+") as f:
        f.write("NEW RUN\n")
    for i in range(args.num_test_runs):
        mean_reward_before, std_reward_before = evaluate_policy(bc_trainer.policy, env, args.eval_episodes)
        with open(f"logs/{args.env_name}/test_results_bc.txt", "a+") as f:
            f.write(f"[Before training] Run {i+1} - Reward: {mean_reward_before} +/- {std_reward_before}\n")
        print(f"[Before training] Run {i+1} - Reward: {mean_reward_before} +/- {std_reward_before}\n")

    #bc_trainer.train(n_epochs=args.epochs)
    agent = train_bc(bc_trainer, data_loader, epochs=args.epochs)
    #save_stable_model(Path("models/bc/"), policy, f"model_{args.epochs}.zip")
    torch.save(agent.policy.state_dict(), f=f"models/bc/{args.env_name}/model_{args.epochs}_{len(data_loader)}_traj.pth")

    for i in range(args.num_test_runs):
        mean_reward_after, std_reward_after = evaluate_policy(bc_trainer.policy, env, args.eval_episodes)
        with open(f"logs/{args.env_name}/test_results_bc.txt", "a+") as f:
            f.write(f"[After training] Run {i+1} - Reward: {mean_reward_after} +/- {std_reward_after}\n")
        print(f"[After training] Run {i+1} - Reward: {mean_reward_after} +/- {std_reward_after}\n")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-env', '--env-name', required=True, type=str)
    arg_parser.add_argument('-d', '--trajectories-path', required=True, type=str)
    arg_parser.add_argument('-e', '--epochs', required=True, type=int)
    arg_parser.add_argument('-t', '--eval-episodes', required=True, type=int)
    arg_parser.add_argument('-b', '--batch-size', default=32, type=int)
    arg_parser.add_argument('-r', '--num-test-runs', default=3, type=int)
    arg_parser.add_argument('-hist', '--history-length', default=16, type=int)
    arg_parser.add_argument('-wb', '--use-wandb', action='store_true')

    args = arg_parser.parse_args()

    main(args)
