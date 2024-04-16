import json
import os

from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np

from tqdm import tqdm
from PIL import Image


def plot_metric_from_log_file(log_file_path, line_to_plot, model, save_fig=True):
    values = []
    with open(log_file_path, 'r', encoding="utf8") as f:
        lines = f.readlines()
        last_value = 0.0
        for l in tqdm(lines):
            if line_to_plot in l:
                l = l.replace(line_to_plot, "")
                l = l.replace("|", "")
                l = l.replace("\n", "")
                l = l.replace(" ", "")
                if float(l) != last_value:
                    values.append(float(l))
                    last_value = float(l)

    fig = plt.figure(figsize=(13, 7))
    ax = plt.Subplot(fig, 111)
    fig.add_subplot(ax)
    ax.plot(range(len(values)), values)
    ax.set_title(line_to_plot.split(sep='/')[-1])
    plt.show()
    if save_fig:
        os.makedirs(f"figures/{model}", exist_ok=True)
        fig.savefig(f"figures/{model}/{line_to_plot.split(sep='/')[-1]}.png")


def process_log_file(file_path):
    timestep = []
    act = []
    il_distribution = []
    il_budget = []
    zip_curr_dist = []
    zip_ref_dist = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for l in lines:
            # Skip episode transition lines
            if not "-----" in l:
                sep_line = l.split(sep="|")
                # Timestep 7 | ZIP action | IL Distrib: [0.089887574, 6.0122158e-05, 0.9100523] | IL Budget: 28 |
                # Current distance: 200.3455352783203 | Reference distance: 0.0 | Followed index: 2731
                timestep.append(int(sep_line[0].replace(" ", "").replace("Timestep", "")))
                # Who acts?
                act.append(sep_line[1])
                # Extract IL distribution
                distr_line = sep_line[2].replace("IL Distrib:", "")
                distr_line = distr_line.replace(" ", "")
                il_distribution.append([float(x) for x in distr_line.strip('][').split(',')])
                # Extract IL budget
                budget_line = sep_line[3].replace("IL Budget:", "")
                budget_line = budget_line.replace(" ", "")
                il_budget.append(int(budget_line))

    return timestep, act, il_distribution, il_budget, zip_curr_dist, zip_ref_dist


def process_log_file_boa(file_path):
    import ast
    timestep = []
    prior = []
    zip_distrib = []
    posterior = []
    act = []
    k = 0
    with open(file_path, "r") as f:
        lines = f.readlines()[1:]
        for i, l in enumerate(lines):
            sep_line = l.split(sep=";")
            #prior;zip_distrib;posterior;action;k
            timestep.append(i)
            prior.append(ast.literal_eval(sep_line[0]))
            zip_distrib.append(ast.literal_eval(sep_line[1]))
            posterior.append(ast.literal_eval(sep_line[2]))
            act.append(int(sep_line[3]))
            k = int(sep_line[4])

    return timestep, prior, zip_distrib, posterior, act, k


def plot_episode_with_stats(agent="bc", width=14, height=12):
    timestep, act, distr, budget, curr_dist, ref_dist = process_log_file(file_path=f"logs/adapted_zip_episode_stats_{agent}.log")
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = cv2.VideoWriter(f"episode_stats_{agent}.avi", fourcc, 10, (width * 100, height * 100))
    for t in tqdm(timestep):
        fig = plt.figure(figsize=(width, height))
        # obs_graph = fig.add_subplot(2, 3, 1)
        agent_distr_graph = fig.add_subplot(2, 2, 1)
        budget_graph = fig.add_subplot(2, 2, 2)
        obs_graph = fig.add_subplot(2, 1, 2)
        obs = plt.imread(f"log_episodes/{agent}/{t}.png")
        obs_graph.imshow(obs)
        obs_graph.axis('off')
        agent_distr_graph.bar(range(len(distr[t])), distr[t])
        agent_distr_graph.set_xticks([0, 1, 2])
        agent_distr_graph.set_xticklabels(["left", "right", "forward"], fontsize=16)
        agent_distr_graph.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        agent_distr_graph.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], fontsize=16)
        agent_distr_graph.text(0.82, 1.12, act[t], fontsize=20)
        for i, v in enumerate(distr[t]):
            agent_distr_graph.text(list(range(len(distr[t])))[i] - 0.18, v + 0.03, f"{v:.2f}", fontsize=20)
        budget_graph.plot(timestep[:t], budget[:t])
        budget_graph.set_xticks(range(0, len(timestep) + int(len(timestep) / 10), int(len(timestep) / 10)))
        budget_graph.set_xticklabels(range(0, len(timestep) + int(len(timestep) / 10), int(len(timestep) / 10)), fontsize=16)
        budget_graph.set_yticks(range(0, 31, 5))
        budget_graph.set_yticklabels(range(0, 31, 5), fontsize=16)
        #distance_graph.plot(timestep[:t], list(range(10)))
        #distance_graph.plot(timestep[:t], list(range(1, 20, 2)))
        fig.suptitle(f"Timestep {t}", fontsize=28)
        plt.tight_layout()
        fig.savefig("tmp.png")
        video.write(cv2.imread("tmp.png"))
        plt.close(fig)
    cv2.destroyAllWindows()
    video.release()


def plot_episode_with_stats_boa(log_dir, width=14, height=12, bar_width=0.5):
    log_files = [name for name in os.listdir(log_dir) if name.endswith(".log")]
    _, env_name, exp_name = log_dir.split(sep='/')
    for ep, log_file in enumerate(log_files):
        timestep, prior, zip_distrib, posterior, act, k = process_log_file_boa(file_path=f"{os.path.join(log_dir, log_file)}")
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        os.makedirs(f"{log_dir}/videos", exist_ok=True)
        video = cv2.VideoWriter(f"{log_dir}/videos/episode_{ep}.avi", fourcc, 10, (width * 100, height * 100))
        obs = np.load(f"recorded/{env_name}/{exp_name}/game_{ep}.npz", allow_pickle=True)["observations"]
        for t in tqdm(timestep):
            fig = plt.figure(figsize=(width, height))
            distrib = fig.add_subplot(1, 2, 1)
            observation = fig.add_subplot(1, 2, 2)
            plt.tight_layout()
            #ax[1] = fig.add_subplot(2, 2, 4)
            observation.axis('off')
            observation.imshow(obs[t, :].transpose(1, 2, 0))
            observation.set_title(f"Timestep {t} - k={k}", fontsize=24)
            for x, (d, offset, c, n, t_off) in enumerate(zip([prior, zip_distrib, posterior], [-bar_width / 2, 0, bar_width / 2], ["tab:blue", "tab:orange", "tab:green"], ["prior", "adaptation", "posterior"], [-0.25, 0.25, 0.5])):
                distrib.bar(list(np.arange(len(d[t])) + offset), d[t], width=bar_width / 2, color=c)
                for i, v in enumerate(d[t]):
                    distrib.text(list(range(len(d[t])))[i] + offset - 0.11, v + 0.03, f"{v:.2f}", fontsize=14)
                distrib.text(x - t_off, 1.1, n, fontsize=14)
                distrib.scatter(x - 0.1 - t_off, 1.115, s=150, c=c)
            distrib.set_xticks(list(range(len(prior[t]))))
            distrib.set_xticklabels(["left", "right", "forward"], fontsize=16)
            distrib.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
            distrib.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], fontsize=16)

            fig.savefig("tmp.png")
            video.write(cv2.imread("tmp.png"))
            plt.close(fig)
        cv2.destroyAllWindows()
        video.release()


def compute_models_stats(results_file_name, task, task_desc):
    colors = ["tab:blue", "tab:red", "tab:green", "gold", "tab:orange", "tab:purple"]
    with open(results_file_name, "r") as f:
        data = json.load(f)
    f.close()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    avgs_succ = []
    std_succ = []
    avgs_dur = []
    std_dur = []
    models = []
    for k in data.keys():
        models.append(k)
        success = data[k]["success"]
        avgs_succ.append(np.mean(success))
        std_succ.append(np.std(success))
        avg_ep_duration = data[k]["avg_ep_duration"]
        avgs_dur.append(np.mean(avg_ep_duration))
        std_dur.append(np.std(avg_ep_duration))

    ax[0].bar(models, avgs_succ, yerr=std_succ, color=colors, capsize=7)
    ax[1].bar(models, avgs_dur, yerr=std_dur, color=colors, capsize=7)
    ax[0].set_xlabel("Model")
    ax[0].set_title("Success percentage (Higher is better)")
    ax[0].set_ylabel("Success rate (%)")
    ax[1].set_xlabel("Model")
    ax[1].set_ylabel("Ep. duration (timesteps)")
    ax[1].set_title("Time for completion (lower is better)")
    fig.suptitle(f"Performance comparison ({task_desc})")
    plt.tight_layout()
    plt.show()
    fig.savefig(f"comparison_{task}.png")


def adapt_trajectories(dir_path, num_traj=-1):
    traj_npz = [x for x in os.listdir(dir_path) if x.endswith(".npz")]
    os.makedirs(f"trajectories/Crafter-v1", exist_ok=True)
    if 0 < num_traj < len(traj_npz):
        traj_npz = traj_npz[:num_traj]
    for el in traj_npz:
        data = dict(np.load(os.path.join(dir_path, el), allow_pickle=True))
        """data["observations"] = data.pop("image")
        data["actions"] = data.pop("action")"""
        np.savez_compressed(f"trajectories/Crafter-v1/{el}", observations=data["image"], actions=data["action"])


def normalize_rlgym_obs(o, eps=1e-9):
    # eps for numerical stability
    ball_pos = (o[:3] - np.min(o[:3])) / (np.max(o[:3]) - np.min(o[:3]) + eps)
    ball_lin_vel = (o[3:6] - np.min(o[3:6])) / (np.max(o[3:6]) - np.min(o[3:6]) + eps)
    ball_ang_vel = (o[6:9] - np.min(o[6:9])) / (np.max(o[6:9]) - np.min(o[6:9]) + eps)
    car_pos = (o[9:12] - np.min(o[9:12])) / (np.max(o[9:12]) - np.min(o[9:12]) + eps)
    car_lin_vel = (o[12:15] - np.min(o[12:15])) / (np.max(o[12:15]) - np.min(o[12:15]) + eps)
    car_ang_vel = (o[15:] - np.min(o[15:])) / (np.max(o[15:]) - np.min(o[15:]) + eps)

    return np.concatenate([ball_pos, ball_lin_vel, ball_ang_vel, car_pos, car_lin_vel, car_ang_vel])


def stack_obs_frames(frames, history_length=4):
    trajectory = []
    frame_buffer = [np.zeros(shape=frames[0].shape) for i in range(history_length)]
    for i in range(frames.shape[0]):
        frame_buffer.pop(0)
        frame_buffer.insert(history_length - 1, frames[i])
        trajectory.insert(i, deepcopy(frame_buffer))

    return np.array(trajectory)


def create_traj_metadata(trajectories_path, num_actions=7):
    counts = np.zeros(shape=(num_actions,))
    for traj_name in tqdm(os.listdir(trajectories_path), desc="Creating trajectories metadata file"):
        with open(f"{trajectories_path}/trajectories_lengths.csv", "a+") as f:
            traj_subdir_path = os.path.join(trajectories_path, f"{traj_name}/no_map/")
            for game_name in os.listdir(traj_subdir_path):
                data = np.load(os.path.join(traj_subdir_path, game_name), allow_pickle=True)
                for x in data["actions"]:
                    counts[x] += 1
                f.write(f"{traj_name}/no_map/{game_name.split(sep='.')[0]},{int(data['actions'].shape[0])}\n")
        f.close()

    print(f"actions: {counts}")


def balance_dataset(trajectories_path, history_length=4):
    counts = np.zeros(shape=(3,))
    trajs = os.listdir(trajectories_path)
    try:
        trajs.remove("trajectories_lengths.csv")
    except Exception:
        pass

    for i in tqdm(trajs, desc="Counting actions"):
        data = np.load(f"{trajectories_path}/{i}", allow_pickle=True)
        for x in data["actions"]:
            counts[x] += 1

    target = int(np.min(counts))
    obs = []
    acts = []
    for i in tqdm(trajs, desc="Balancing dataset"):
        data = np.load(f"{trajectories_path}/{i}", allow_pickle=True)
        obs.append(stack_obs_frames(data["observations"], history_length))
        acts.append(data["actions"])

    obs = np.concatenate(obs, axis=0)
    acts = np.concatenate(acts, axis=0)

    new_obs = []
    new_acts = []
    for x in range(counts.shape[0]):
        idx = np.argwhere(acts == x)
        if counts[x] > target:
            idx = idx[0: target]
        new_obs.append(obs[idx])
        new_acts.append(acts[idx])

    new_obs = np.concatenate(new_obs, axis=0)
    new_acts = np.concatenate(new_acts, axis=0)

    indices = np.arange(len(new_acts))
    np.random.shuffle(indices)
    os.makedirs(f"balanced_trajectories/{trajectories_path.split(sep='/')[-1]}/", exist_ok=True)
    np.savez_compressed(f"balanced_trajectories/{trajectories_path.split(sep='/')[-1]}/0.npz", observations=new_obs[indices], actions=new_acts[indices])


def result_file_to_json(file_path, env, return_json=False):
    dir_path = file_path.replace(file_path.split(sep="/")[-1], "")
    with open(file_path, "r") as f:
        lines = f.readlines()
        f.close()

    if len(lines) > 1:
        results = {}
        num_steps = []
        success = []
        ep_duration = []
        reward = []
        std_steps = []
        std_ep_dur = []
        std_reward = []
        file_name = file_path.split(sep="/")[-1].split(sep=".")[0]
        for l in lines:
            if l.startswith("NEW RUN"):
                if len(num_steps) > 0:
                    run = {
                        "avg_num_steps": num_steps,
                        "success": success,
                        "avg_episode_duration": ep_duration,
                        "avg_reward": reward
                    }
                    results[run_title] = run
                run_title = l.replace("NEW RUN - ", "").replace("\n", "")
                num_steps = []
                success = []
                ep_duration = []
                reward = []
                std_steps = []
                std_ep_dur = []
                std_reward = []
            else:
                strings = l.split(sep=" | ")
                for s in strings:
                    s = s.lower()
                    if "avg. number of steps:" in s:
                        s = s.replace("avg. number of steps: ", "")
                        mean, std = s.split(" +/- ")
                        mean = float(mean)
                        num_steps.append(mean)
                        std_steps.append(float(std))
                    elif "success percentage:" in s:
                        s = s.replace("success percentage: ", "")
                        s = s.replace("%", "")
                        success.append(float(s))
                    elif "avg. episode duration" in s:
                        s = s.replace("avg. episode duration: ", "")
                        mean, std = s.split(" +/- ")
                        mean = float(mean)
                        ep_duration.append(mean)
                        std_ep_dur.append(float(std))
                    elif "avg. reward" in s:
                        s = s.replace("avg. reward: ", "")
                        mean, std = s.split(" +/- ")
                        mean = float(mean)
                        reward.append(mean)
                        std_reward.append(float(std))
        run = {
            "avg_num_steps": num_steps,
            "success": success,
            "avg_episode_duration": ep_duration,
            "avg_reward": reward
        }
        results[run_title] = run

        if return_json:
            return results
        else:
            os.makedirs(f"{dir_path}/json_parsed/{env}", exist_ok=True)
            with open(f"{dir_path}/json_parsed/{env}/{file_name}.json", "w+") as f:
                json.dump(results, f)
                f.close()
            return None


def generate_json_files(dir_path, envs):
    for e in envs:
        env_name = f"MiniWorld-{e}-v0"
        for fp in os.listdir(f"{dir_path}/{env_name}"):
            result_file_to_json(f"{dir_path}/{env_name}/{fp}", env=env_name)


def build_image_from_json(metric: str, allowed_models: list):
    colors = ["tab:grey", "gold", "tab:blue", "tab:green", "tab:red"]
    envs = os.listdir("results/json_parsed")
    plt.tight_layout()
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 8))
    plt.subplots_adjust(hspace=0.2,
                        top=0.95,
                        left=0.05,
                        right=0.99,
                        bottom=0.02
    )
    for i, e in enumerate(envs):
        agents = {}
        agents_result_files = os.listdir(f"results/json_parsed/{e}")
        for results in agents_result_files:
            with open(f"results/json_parsed/{e}/{results}", "r") as f:
                data = dict(json.load(f))
                for k, v in data.items():
                    for m in allowed_models:
                        if m in k:
                            agents[results.split(sep=".")[0].split(sep="_")[0]] = v
                            break
        # Build the plot
        assert metric in ["duration", "success", "reward"], "Unknown metric."
        a = agents.keys()
        a = [list(a)[i] for i in [3, 4, 0, 1, 2]]
        if metric == "duration":
            means = [agents[k]["avg_num_steps"] for k in a]
            std = [agents[k]["std_num_steps"] for k in a]
            if "CollectHealth" in e:
                yticks = [0, 20, 40, 60, 80, 100]
            elif any(["Hallway" in e, "OneRoom" in e, "Sidewalk" in e, "TMaze" in e, "YMaze" in e]):
                yticks = [0, 100, 200, 300, 400, 500]
            else:
                yticks = [0, 200, 400, 600, 800, 1000]
        elif metric == "success":
            means = [agents[k]["success"] for k in a]
            std = [np.std(x) for x in means]
            yticks = [0, 20, 40, 60, 80, 100]
        else:
            means = [agents[k]["avg_reward"] for k in a]
            std = [agents[k]["std_reward"] for k in a]
            if "CollectHealth" in e:
                yticks = [0, 10, 20, 30, 40, 50]
            elif "PutNext" in e:
                yticks = [0, 0.05, 0.10, 0.15, 0.2, 0.25]
            else:
                yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

        pos = list(range(len(agents.keys())))
        violins = ax[int(i / 5), i % 5].violinplot(means, pos, showmeans=True, showextrema=True, showmedians=True, widths=1)
        #ax[int(i / 5), i % 5].bar(pos, [np.mean(x) for x in means], yerr=[np.mean(np.sqrt(x)) for x in std], color=colors, capsize=7)
        ax[int(i / 5), i % 5].set_title(f"{e.split(sep='-')[1]}")
        ax[int(i / 5), i % 5].set_xticks(pos)
        ax[int(i / 5), i % 5].set_xticklabels(a, rotation=45)
        ax[int(i / 5), i % 5].set_yticks(yticks)
        ax[int(i / 5), i % 5].set_yticklabels(yticks)
        ax[0, 0].set_ylabel(metric)
        ax[1, 0].set_ylabel(metric)
        ax[1, 0].legend(a, loc="lower left")
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i, j].get_xaxis().set_visible(False)
        # Colors
        for pc, color in zip(violins['bodies'], colors):
            pc.set_facecolor(color)
        violins['cmedians'].set_colors(colors)
        violins['cbars'].set_colors(colors)
        violins['cmaxes'].set_colors(colors)
        violins['cmins'].set_colors(colors)
        violins['cmeans'].set_colors(colors)

    #fig.suptitle(f"Results - {metric}")
    fig.savefig(f"figures/results/{metric}.png")
    fig.savefig(f"figures/results/{metric}.pdf")


def compute_human_normalized_score_for_expert(env_name, max_steps):
    human_scores = {}
    try:
        traj_lengths = np.genfromtxt(f"trajectories/MiniWorld-{env_name}-v0/trajectories_lengths.csv", delimiter=",")
    except Exception:
        create_traj_metadata(f"trajectories/MiniWorld-{env_name}-v0/")
        traj_lengths = np.genfromtxt(f"trajectories/MiniWorld-{env_name}-v0/trajectories_lengths.csv", delimiter=",")

    if env_name == "CollectHealth":
        scores = 2.0 * traj_lengths[:, 1] - 100
    else:
        scores = 1 - 0.2 * (traj_lengths[:, 1] / max_steps)

    human_scores["avg_num_steps"] = np.mean(traj_lengths[:, 1])
    human_scores["std_num_steps"] = np.std(traj_lengths[:, 1])
    human_scores["avg_reward"] = np.mean(scores)
    human_scores["std_reward"] = np.std(scores)
    human_scores["success"] = 100.0

    return human_scores


def create_image_human_normalized(human_scores: dict, metric: str, allowed_models: list):
    assert metric in ["duration", "success", "reward"], "Unknown metric."

    plt.tight_layout()
    #rc('text', usetex=True)
    #rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

    if metric == "duration":
        results_key_avg = "avg_num_steps"
    elif metric == "reward":
        plt.subplots_adjust(hspace=0.5)
        results_key_avg = "avg_reward"
    else:
        results_key_avg = "success"
    env = human_scores.keys()
    data_avg = [[] for _ in env]
    data_std = [[] for _ in env]
    for i, e in enumerate(env):
        agents_result_files = os.listdir(f"results/json_parsed/MiniWorld-{e}-v0")
        agents_result_files.remove("random_test_results.json")

        for j, results in enumerate(agents_result_files):
            with open(f"results/json_parsed/MiniWorld-{e}-v0/{results}", "r") as f:
                data = dict(json.load(f))
                for k, v in data.items():
                    for m in allowed_models:
                        if m in k:
                            if metric in ["duration", "reward"]:
                                data_avg[i].append(np.mean(v[results_key_avg]))
                                data_std[i].append(np.std(v[results_key_avg]))
                            else:
                                data_avg[i].append((np.mean(v[results_key_avg]) / human_scores[e][results_key_avg]) * 100)
                                data_std[i].append((np.std(v[results_key_avg]) / human_scores[e][results_key_avg]) * 100)
        data_avg[i] = [data_avg[i][k] for k in [4, 5, 0, 3, 1, 2]]
        data_std[i] = [data_std[i][k] for k in [4, 5, 0, 3, 1, 2]]

    """a = [k.split(sep="_")[0] for k in agents_result_files]
    a = [a[k] for k in [3, 4, 0, 1, 2]]"""

    if metric == "duration":
        xticks = [0, 200, 400, 600, 800, 1000, 1200]
        x_label = "Timestep"
    elif metric == "reward":
        xticks = [-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        x_label = "Reward"
    else:
        xticks = [-10, 0, 20, 40, 60, 80, 100, 120]
        x_label = "Human performance (%)"
    colors = ["silver", "silver", "tab:blue", "tab:blue", "tab:red", "tab:red"]
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30, 8))
    for i, e in enumerate(env):
        ax[int(i / 5), i % 5].barh(y=list(range(6)), width=data_avg[i], xerr=data_std[i], capsize=5, color=colors, zorder=0, hatch=["", "", "", "//", "", "//"])
        ax[int(i / 5), i % 5].grid(axis="x", zorder=-1)
        ax[int(i / 5), i % 5].set_yticks(list(range(6)))
        ax[int(i / 5), i % 5].set_yticklabels(["PPO", "ZIP", "BC", "BOA+BC", "GAIL", "BOA+GAIL"], fontsize=16)
        ax[int(i / 5), i % 5].set_xticks(xticks)
        ax[int(i / 5), i % 5].set_xticklabels(xticks, fontsize=15)
        ax[int(i / 5), i % 5].set_title(e, fontsize=19)

    if metric == "reward":
        ax[0, 0].set_xticks([-10, 0, 20, 40, 60, 80, 100, 120])
        ax[0, 0].set_xticklabels([-10, 0, 20, 40, 60, 80, 100, 120], fontsize=15)
    elif metric == "duration":
        ax[0, 0].set_xlim(0, 120)
        ax[0, 0].set_xticks([0, 20, 40, 60, 80, 100, 120])
        ax[0, 0].set_xticklabels([0, 20, 40, 60, 80, 100, 120], fontsize=15)
    else:
        ax[0, 0].set_xticklabels([])
    for x in range(1, 5):
        ax[0, x].set_xticklabels([])
        ax[0, x].set_yticklabels([])
        ax[1, x].set_yticklabels([])

    fig.text(0.5, 0.02, x_label, ha='center', va='center', fontsize=18)
    fig.text(0.07, 0.5, 'Agent', ha='center', va='center', rotation='vertical', fontsize=18)

    fig.savefig(f"figures/results/{metric}_norm.png", bbox_inches='tight')
    fig.savefig(f"figures/results/{metric}_norm.pdf", bbox_inches='tight')


def plot_k_test(dir_path, metric="success"):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 6))
    plt.subplots_adjust(hspace=0.2,
                        top=0.95,
                        left=0.1,
                        right=0.99
                        )
    """all_k = []
    handles = []"""
    handles = []
    envs = [x.split(sep=".")[0].replace("gail_", "") for x in os.listdir(dir_path) if "gail" in x]
    for i, env_name in enumerate(envs):
        relevant_files = [x for x in os.listdir(dir_path) if env_name in x]
        for file_path in relevant_files:
            data = result_file_to_json(f"{dir_path}/{file_path}", env=env_name, return_json=True)
            mean_values = []
            std_values = []
            k_values = []
            for key, value in data.items():
                k = int(key.split(sep=" - ")[-1].split(sep="=")[-1])
                if metric == "reward":
                    v = value["avg_reward"]
                elif metric == "duration":
                    v = value["avg_num_steps"]
                else:
                    v = value["success"]
                mean = np.mean(v)
                std = np.std(v)
                k_values.append(k)
                mean_values.append(mean)
                std_values.append(std)
            c = "tab:red" if "gail" in file_path else "tab:blue"
            line = axes[int(i / 5), i % 5].plot(k_values, mean_values, color=c)
            if i == 0:
                handles.append(line[0])
            axes[int(i / 5), i % 5].fill_between(k_values, y1=np.array(mean_values) - np.array(std_values),
                            y2=np.array(mean_values) + np.array(std_values), alpha=0.3, color=c)
            axes[int(i / 5), i % 5].scatter(k_values[np.argmax(mean_values)], np.max(mean_values), s=75, marker="*", edgecolor="black",
                       zorder=3, color=c)
        axes[int(i / 5), i % 5].set_title(env_name.split(sep="-")[1])
        """axes[int(i / 5), i % 5].set_yticks([0, 20, 40, 60, 80, 100, 120])
        axes[int(i / 5), i % 5].set_yticklabels([0, 20, 40, 60, 80, 100, 120])"""

    for j in range(5):
        axes[0, j].set_xticklabels([])
    """for j in range(1, 5):
        axes[0, j].set_yticklabels([])
        axes[1, j].set_yticklabels([])"""
    axes[0, 3].legend(handles, ["BOA+BC", "BOA+GAIL"])

    """cmap = matplotlib.cm.get_cmap("tab10")
    col = [cmap(i) for i in np.arange(0, 1, 0.1)]
    for agent, ax in zip(["gail", "bc"], axes):
        test_files = [x for x in os.listdir(dir_path) if agent in x]
        for i, (fp, c) in enumerate(zip(test_files, col)):
            env_name = fp.split(sep="_")[1]
            env_name = env_name.split(sep=".")[0]
            envs.append(env_name.split(sep="-")[1])
            data = result_file_to_json(f"{dir_path}/{fp}", env=env_name, return_json=True)
            mean_values = []
            std_values = []
            k_values = []
            for key, value in data.items():
                k = int(key.split(sep=" - ")[-1].split(sep="=")[-1])
                if metric == "reward":
                    v = value["avg_reward"]
                elif metric == "duration":
                    v = value["avg_num_steps"]
                else:
                    v = value["success"]
                mean = np.mean(v)
                std = np.std(v)
                k_values.append(k)
                mean_values.append(mean)
                std_values.append(std)
            if i == 0 and agent == "gail":
                all_k = k_values
            line = ax.plot(k_values, mean_values, color=c)
            if agent == "gail":
                handles.append(line[0])
            ax.fill_between(k_values, y1=np.array(mean_values) - np.array(std_values), y2=np.array(mean_values) + np.array(std_values), alpha=0.3, color=c)
            ax.scatter(k_values[np.argmax(mean_values)], np.max(mean_values), s=75, marker="*", edgecolor="black", zorder=3, color=c)
            print(f"{env_name} - best k: {k_values[np.argmax(mean_values)]} - SUCCESS RATE: {mean_values[np.argmax(mean_values)]:.2f} +/- {std_values[np.argmax(mean_values)]:.2f}")
        ax.set_xticks(all_k)
        ax.set_xticklabels(all_k)
    fig.text(0.5, 0.02, "Number of retrieved examples", ha='center', va='center', fontsize=18)
    axes[0].set_ylabel("Avg. success rate (%)", fontsize=16)
    axes[0].legend(handles=handles, labels=envs, markerscale=0.3, fontsize=8, bbox_to_anchor=(0.95, 0.74), loc="center right")"""

    """fig.text(0.085, 0.85, "A", ha='center', va='center', fontsize=48)
    fig.text(0.505, 0.85, "B", ha='center', va='center', fontsize=48)"""
    plt.show()

    fig.text(0.5, 0.02, "Retrieved examples", ha='center', va='center', fontsize=18)
    fig.text(0.07, 0.5, 'Success rate (%)', ha='center', va='center', rotation='vertical', fontsize=18)

    fig.savefig(f"figures/results/k_test.png", bbox_inches='tight')
    fig.savefig(f"figures/results/k_test.pdf", bbox_inches='tight')


def plot_num_traj_ablation(file_paths, metric="success"):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    handles = []
    for file_path in file_paths:
        data = result_file_to_json(file_path, env="MazeS3", return_json=True)
        mean_values = []
        std_values = []
        k_values = []
        for key, value in data.items():
            k = int(key.split(sep=" - ")[-1].split(sep="=")[-1])
            if metric == "reward":
                v = value["avg_reward"]
            elif metric == "duration":
                v = value["avg_num_steps"]
            else:
                v = value["success"]
            mean = np.mean(v)
            std = np.std(v)
            k_values.append(k)
            mean_values.append(mean)
            std_values.append(std)

        c = "tab:blue" if "bc" in file_path else "tab:red"
        line = ax.plot(k_values, mean_values, color=c)
        handles.append(line[0])
        ax.fill_between(k_values, y1=np.array(mean_values) - np.array(std_values),
                        y2=np.array(mean_values) + np.array(std_values), alpha=0.3, color=c)
        ax.scatter(k_values[np.argmax(mean_values)], np.max(mean_values), s=75, marker="*", edgecolor="black", zorder=3, color=c)
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    ax.set_xticklabels([0, 30, 60, 90, 120, 150])
    ax.set_yticks([20, 30, 40, 50, 60, 70])
    ax.set_yticklabels([20, 30, 40, 50, 60, 70])
    ax.set_xlabel("Expert trajectories encoded", fontsize=14)
    ax.set_ylabel("Avg. success rate (%)", fontsize=14)
    ax.legend(handles, ["BOA+GAIL", "BOA+BC"], bbox_to_anchor=(0.9, 0.9), loc="center right")

    plt.show()

    fig.savefig(f"figures/results/num_traj_test.png", bbox_inches='tight')
    fig.savefig(f"figures/results/num_traj_test.pdf", bbox_inches='tight')


def env_images():
    for env in [x for x in os.listdir("trajectories") if "MiniWorld" in x]:
        traj = np.random.randint(0, 19)
        data = np.load(f"trajectories/{env}/{traj}.npz", allow_pickle=True)
        idx = np.random.randint(0, data["observations"].shape[0] - 1)
        img = Image.fromarray(data["observations"][idx])
        img.save(f"figures/envs/{env}.png")


if __name__ == '__main__':

    """allowed_models = [
        "model_100_ep_20_traj_4_hist_len.zip",
        "model_80_ep_20_traj_4_hist_len.zip",
        "model_50_ep_20_traj_4_hist_len.pth",
        "model_250_ep_20_traj_4_hist_len.pth",
        "1000000_ts_4_hist_len.pth",
    ]
    env_names = [
        "CollectHealth",
        "FourRooms",
        "Hallway",
        "MazeS3",
        "OneRoom",
        "PutNext",
        "Sidewalk",
        "TMaze",
        "WallGap",
        "YMaze"
    ]
    env_steps = [
        1000,
        1000,
        500,
        1000,
        500,
        1000,
        500,
        500,
        1000,
        500
    ]

    results = {}
    for env in [x for x in os.listdir("results/") if "MiniWorld" in x]:
        results[env] = []
        for file_path in os.listdir(f"results/{env}"):
            results[env].append(result_file_to_json(f"results/{env}/{file_path}", env=env, return_json=True))

    scores = {}
    for name, steps in zip(env_names, env_steps):
        human_score = compute_human_normalized_score_for_expert(name, steps)
        scores[name] = human_score

    create_image_human_normalized(scores, metric="duration", allowed_models=allowed_models)"""

    #plot_k_test("results/k_tests", metric="success")
    plot_num_traj_ablation(["results/num_trajectories_test_gail.txt", "results/num_trajectories_test_bc.txt"])
