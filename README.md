# MiniCrawl: a Minimalistic Dungeon Crawler for Reinforcement and Imitation Learning Research

MiniCrawl is a dungeon crawler built on top of [MiniWorld](https://github.com/Farama-Foundation/Miniworld). The library 
uses and extends methods from `MiniWorld`, but _does not_ change existing code of the base library in any way. 

**Contents**

- Introduction
- Installation
- Usage
- Environment
- Known Bugs

## Introduction
MiniCrawl is a minimalistic dungeon crawler-like 3D environment for Reinforcement and Imitation Learning research. 
The dungeon is structured as a sequence of levels. When a level is cleared, the environment automatically creates a
new one.

Being built on top of MiniWorld, the environment inherits base library's features and limitations.
## Installation
### Requirements
- Python 3.7+
- [MiniWorld](https://github.com/Farama-Foundation/Miniworld)
- [keyboard](https://pypi.org/project/keyboard/)

MiniCrawl can be installed via pip:

    pip install minicrawl

alternatively, this repo can be cloned using `git clone`. Then, execute

    cd minicrawl && python3 -m pip install -e .

to install the library.
## Usage
A ready-to-use, human-controlled version of the environment is provided in `main.py`. If you want to collect your own
data following the custom dataset format, simply run `collect_data.py`.

To create a new instance of environment, simply use:

    env = gym.make("MiniCrawl-DungeonCrawlerEnv-v0", render_mode="human")

additional options are `render_map: bool` to dynamically render the current floor map and `max_level: int` to specify
the maximum level an agent can reach.

## Environment
The dungeon is composed of a number of levels that can be limited or unlimited. Each level can be either a `DungeonFloor` 
or a `BossStageFloor`. For Reinforcement Learning purposes, each floor attributes a reward upon completion. Additionally,
a total reward is computed as the sum of the floor rewards at the end of a run.

### Dungeon Floor
A dungeon floor consists of a precedurally-generated ensemble of rooms and corridors. The goal of a dungeon floor is to
navigate the maze and retrieve a yellow key. The agent spawns in a random room of the maze.

Dungeon floors increase in size as an agent goes deeper in the dungeon. The **Reward** for a dungeon floor is
+(1 - 0.2 * (`step_count` / `max_episode_steps`)) * `current_level` when the level is cleared, zero otherwise.

### Boss Stage
Every `boss_stage_freq` dungeon floors, a `BossStageFloor` is created. Boss stages are used to diversify the skills that
an agent needs to solve the environment. Each boss stage features a prompt written on a wall, using MiniWorld's `TextFrame`.
Currently, there are three types of `BossStageFloor`:
- `PutNextBossStageFloor`: inspired from MiniWorld's `MiniWorld-PutNext-v0` environment. An agent spawns in the middle
of a room featuring four colored cubes, one per each corner. The prompt on the wall randomly selects two cubes that have
to be placed close to each other. When the mission is accomplished, the agent advances to the next level. **Reward**:
+(1 - 0.2 * (`step_count` / `max_episode_steps`)) * `current_level` upon completion of the level, zero for failure.
- `PickUpBossStageFloor`: inspired from MiniWorld's `MiniWorld-Sign-v0` environment. The agent is placed in a corner of
a big room filled with objects. The prompt specifies which object an agent must collect. Upon completion of the task, 
the agent advances to the next level. Picking up a different object leads to failure. **Reward**:
+(1 - 0.2 * (`step_count` / `max_episode_steps`)) * `current_level` upon completion of the level, -10 for picking up 
the wrong object.
- `AvoidObstaclesBossStageFloor`: inspired from [MiniGrid](https://github.com/Farama-Foundation/Minigrid)'s 
`MiniGrid-Dynamic-Obstacles-NxN-v0` environment, the agent is placed in a corner of a big room. At the opposite corner,
a yellow key is spawned. The room is filled with moving blue balls. Similar to [SuperHot](https://superhotgame.com/), the
obstacles move only when the agent move. The level is cleared when the agent successfully traverses the room and
collects the key. Colliding with an obstacle makes the agent fail. **Reward**: 
+(1 - 0.2 * (`step_count` / `max_episode_steps`)) * `current_level` when the key is collected, -10 for colliding with 
an obstacle.

## Known Bugs
- Please launch the game in a folder you own (i.e. no root permissions needed). Otherwise, the game will return an error.
- If the game window opens but the agent does not move, click on the
  terminal window you used to launch the experiment. The game only record
  keys pressed within that terminal window.
- If you get stuck on a wall, just take some steps back before proceeding.

If you encounter any unknown bug, please report it in the **issue** section.

# Citation