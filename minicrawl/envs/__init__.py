import gymnasium as gym

from minicrawl.envs.testdungeon import TestDungeon
from minicrawl.envs.dungeon_floor import DungeonFloorEnv


__all__ = [
    "TestDungeon"
]

gym.register(
    id="MiniCrawl-TestDungeon-v0",
    entry_point="minicrawl.envs.testdungeon:TestDungeon"
)

gym.register(
    id="MiniCrawl-DungeonCrawlerEnv-v0",
    entry_point="minicrawl.minicrawl:MiniCrawlEnv"
)

gym.register(
    id="MiniCrawl-DungeonFloorEnv-v0",
    entry_point="minicrawl.envs.dungeon_floor:DungeonFloorEnv"
)

gym.register(
    id="MiniCrawl-PutNextBossStageEnv-v0",
    entry_point="minicrawl.envs.boss_stage:PutNextBossStageEnv"
)
