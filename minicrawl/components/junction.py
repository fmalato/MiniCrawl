import numpy as np

from miniworld.miniworld import Room, DEFAULT_WALL_HEIGHT


class JunctionRoom(Room):
    def __init__(
            self,
            position,
            cell_size=6,
            edge_size=3,
            wall_height=DEFAULT_WALL_HEIGHT,
            floor_tex="floor_tiles_bw",
            wall_tex="concrete",
            ceil_text="concrete_tiles",
            no_ceiling=False
    ):
        outline = np.array([
            # East wall
            [(position[1] + 1) * cell_size - ((cell_size - edge_size) / 2), (position[0] + 1) * cell_size - ((cell_size - edge_size) / 2)],
            # North wall
            [(position[1] + 1) * cell_size - ((cell_size - edge_size) / 2), position[0] * cell_size + ((cell_size - edge_size) / 2)],
            # West wall
            [position[1] * cell_size + ((cell_size - edge_size) / 2), position[0] * cell_size + ((cell_size - edge_size) / 2)],
            # South wall
            [position[1] * cell_size + ((cell_size - edge_size) / 2), (position[0] + 1) * cell_size - ((cell_size - edge_size) / 2)],
        ])
        super().__init__(outline, wall_height, floor_tex, wall_tex, ceil_text, no_ceiling)


if __name__ == '__main__':
    r = JunctionRoom(position=(1, 0), cell_size=12, edge_size=3)
