import gymnasium as gym

from minicrawl.envs.testdungeon import TestDungeon


__all__ = [
    "TestDungeon"
]

gym.register(
    id="MiniCrawl-TestDungeon-v0",
    entry_point="minicrawl.envs.testdungeon:TestDungeon"
)
