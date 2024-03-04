import argparse
import keyboard

import gymnasium as gym
import miniworld


def map_action(key):
    action = 6
    if key == "a":
        action = 0
    elif key == "d":
        action = 1
    elif key == "w":
        action = 2
    elif key == "s":
        action = 3
    elif key == "e":
        action = 4
    elif key == "q":
        action = 5

    return action


def record(args):
    env = gym.make("MiniWorld-ThreeRooms-v0", render_mode="human")

    for i in range(args.num_games):
        observation, info = env.reset()
        terminated = False
        observations = []
        actions = []
        while not terminated:
            key = keyboard.read_key(suppress=True)
            action = map_action(key)
            observations.append(observation)
            actions.append(action)
            observation, reward, terminated, truncated, info = env.step(action)
            env.render_top_view()
            env.render()

    env.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-n', '--num-games', default=1, type=int)

    args = arg_parser.parse_args()

    record(args)
