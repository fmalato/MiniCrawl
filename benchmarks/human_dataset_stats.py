import os

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm


DATA_PATH = "data/"


def extract_multimodal(runs, map_shown=True):
    dists = []
    for r in runs:
        traj_path = os.path.join(DATA_PATH, r, "map" if map_shown else "no_map")
        for g in os.listdir(traj_path):
            data_no_map = np.load(os.path.join(traj_path, g), allow_pickle=True)
            actions = data_no_map["actions"]
            dists.append(norm.pdf(ls, np.mean(actions), np.std(actions)))
    multimodal_norm = deepcopy(dists[0])
    for d in dists[1:]:
        multimodal_norm += d
    multimodal_norm = multimodal_norm / len(dists)

    mins = []
    maxs = []
    matrix_dists = np.array(dists)
    for i in range(matrix_dists.shape[1]):
        mins.append(np.min(matrix_dists[:, i]))
        maxs.append(np.max(matrix_dists[:, i]))

    return multimodal_norm, np.array(mins), np.array(maxs)

def plot_map_occupancy(maps_images, ax):
    RED = np.array([255, 0, 0])
    background_image = maps_images[0, :]
    for img in maps_images[1: maps_images.shape[0] - 2, :, :, :]:
        player_position_x, player_position_y = np.where(np.all(img == RED, axis=-1))
        background_image[player_position_x, player_position_y, 2] = 255
    ax.imshow(background_image)
    ax.axis('off')


"""
observations
actions
level_map
rewards
new_level
has_reached_maximum
total_reward
"""


if __name__ == '__main__':
    #################### PLOT ACTION DISTRIBUTION ####################
    ls = np.linspace(0, 7, 1000)
    runs = os.listdir(DATA_PATH)
    try:
        runs.remove("trajectories_lengths.csv")
    except ValueError:
        pass
    action_distrib_fig, action_distrib_ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    # No map
    distrib, min_values, max_values = extract_multimodal(runs, map_shown=False)
    no_map_handle = action_distrib_ax.plot(ls, distrib, color="tab:red")[0]
    action_distrib_ax.fill_between(ls, min_values, max_values, color="tab:red", alpha=0.3)
    # Map
    distrib, min_values, max_values = extract_multimodal(runs, map_shown=True)
    map_handle = action_distrib_ax.plot(ls, distrib, color="tab:blue")[0]
    action_distrib_ax.fill_between(ls, min_values, max_values, color="tab:blue", alpha=0.3)
    # Style
    action_distrib_ax.legend(handles=[no_map_handle, map_handle], labels=["No map", "Map"])
    plt.tight_layout()
    plt.show()

    #################### PLOT OCCUPANCY MAP ####################
    """for r in runs:
        traj_path = os.path.join(DATA_PATH, r, "map")
        for g in os.listdir(traj_path):
            data_no_map = np.load(os.path.join(traj_path, g), allow_pickle=True)
            level_maps = data_no_map["level_map"]
            level_finished_indices = np.argwhere(data_no_map["new_level"])
            plot_map_occupancy(level_maps[int(level_finished_indices[0]):int(level_finished_indices[1]), :, :, :])
            for i in range(level_finished_indices.shape[0] - 1):
                plot_map_occupancy(level_maps[level_finished_indices[i]:level_finished_indices[i-1], :, :, :])"""
    # No map
    occupancy_map_fig, occupancy_map_ax = plt.subplots(nrows=2, ncols=10, figsize=(30, 6))
    n = 0
    for k in range(3, 8, 1):
        data_no_map = np.load(f"map_plot/no_map_{k}.npz", allow_pickle=True)
        level_maps = data_no_map["level_map"]
        level_finished_indices = np.argwhere(data_no_map["new_level"])
        level_finished_indices = np.insert(level_finished_indices, 0, 0)
        for i in range(2):
            try:
                # TODO: something wrong with this
                plot_map_occupancy(level_maps[int(level_finished_indices[i]):int(level_finished_indices[i + 1]), :, :, :], occupancy_map_ax[0, n])
                n += 1
            except IndexError:
                continue
        # Map
        n -= 2
        data_no_map = np.load(f"map_plot/map_shown_{k}.npz", allow_pickle=True)
        level_maps = data_no_map["level_map"]
        level_finished_indices = np.argwhere(data_no_map["new_level"])
        level_finished_indices = np.insert(level_finished_indices, 0, 0)
        for i in range(2):
            try:
                plot_map_occupancy(level_maps[int(level_finished_indices[i]):int(level_finished_indices[i + 1]), :, :, :], occupancy_map_ax[1, n])
                n += 1
            except IndexError:
                continue

    plt.show()
