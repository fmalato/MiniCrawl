import numpy as np
import skvideo.io
import ffmpeg

from PIL import Image, ImageDraw
from tqdm import tqdm


def clean_data(act_f, obs_f):
    acts_t = act_f.readlines()
    obs_t = obs_f.readlines()
    key_presses = []
    for i, act_l in enumerate(acts_t):
        try:
            t, k = act_l.split(sep=",")
            k = k.replace("\n", "")
        except ValueError:
            t = float(act_l.replace(",\n", ""))
            k = ""
        acts_t[i] = float(t)
        key_presses.append(k)
    for i, obs_l in enumerate(obs_t):
        obs_t[i] = float(obs_l.replace("\n", ""))

    return acts_t, obs_t, key_presses


def sync_obs_with_acts(acts_t, obs_t, return_array=False, folder_name=None):
    mapped_acts = []
    for i, obs in enumerate(obs_t):
        diff = [np.abs(obs - act) for act in acts_t]
        mapped_acts.append((i, np.argmin(diff)))

    if folder_name is not None:
        with open(f"videos/{folder_name}/sync_acts.csv", "a+") as f:
            for o, a in mapped_acts:
                f.write(f"{obs_t[o]},{acts_t[a]}\n")
        f.close()

    if return_array:
        return mapped_acts
    else:
        return None


def sync_video_demo(key_presses, mapped_acts, folder_name):
    videodata = skvideo.io.vread(f"videos/{folder_name}/video.mp4")
    frames = []
    for i, (o, a) in tqdm(enumerate(mapped_acts)):
        frame = Image.fromarray(videodata[i, :])
        ImageDraw.Draw(frame).text((5, 85), key_presses[a], (0, 255, 0))
        frames.append(np.array(frame))
    frames = np.asarray(frames)

    n, height, width, channels = frames.shape
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(f"videos/{folder_name}/video_sync.mp4", pix_fmt='yuv420p', vcodec='libx264')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for frame in frames:
        process.stdin.write(
            frame
            .astype(np.uint8)
            .tobytes()
        )
    process.stdin.close()
    process.wait()


if __name__ == '__main__':
    folder_name = "pwR0RXcSbjYtzLSB3kEkvEkW"
    with open(f"videos/{folder_name}/actions.csv", "r") as act_f, open(f"videos/{folder_name}/obs_timestamps.csv", "r") as obs_f:
        acts_t, obs_t, key_presses = clean_data(act_f, obs_f)
    mapped_acts = sync_obs_with_acts(acts_t, obs_t, return_array=True, folder_name=folder_name)
    sync_video_demo(key_presses, mapped_acts, folder_name)
