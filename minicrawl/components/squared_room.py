import numpy as np

from miniworld.miniworld import Room, DEFAULT_WALL_HEIGHT


class SquaredRoom(Room):
    def __init__(
            self,
            position,
            edge_size=6,
            wall_height=DEFAULT_WALL_HEIGHT,
            floor_tex="floor_tiles_bw",
            wall_tex="concrete",
            ceil_text="concrete_tiles",
            no_ceiling=False
    ):
        outline = np.array([
            # East wall
            [(position[1] + 1) * edge_size, (position[0] + 1) * edge_size],
            # North wall
            [(position[1] + 1) * edge_size, position[0] * edge_size],
            # West wall
            [position[1] * edge_size, position[0] * edge_size],
            # South wall
            [position[1] * edge_size, (position[0] + 1) * edge_size],
        ])
        super().__init__(outline, wall_height, floor_tex, wall_tex, ceil_text, no_ceiling)


if __name__ == '__main__':
    r = SquaredRoom(position=(0, 2), edge_size=12)