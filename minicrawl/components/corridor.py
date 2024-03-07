import numpy as np

from miniworld.miniworld import Room, DEFAULT_WALL_HEIGHT


class Corridor(Room):
    def __init__(
            self,
            position,
            orientation,
            cell_size,
            junction_size,
            wall_height=DEFAULT_WALL_HEIGHT,
            floor_tex="floor_tiles_bw",
            wall_tex="concrete",
            ceil_text="concrete_tiles",
            no_ceiling=False
    ):
        assert orientation in ["north", "south", "east", "west"], "Unknown orientation."

        self.min_x = None
        self.max_x = None
        self.min_z = None
        self.max_z = None
        outline = self._define_outline(position, orientation, cell_size, junction_size)
        super().__init__(outline, wall_height, floor_tex, wall_tex, ceil_text, no_ceiling)

    def _define_outline(self, position, orientation, cell_size, junction_size):
        if orientation == "east":
            self.min_x = position[1] * cell_size + (3 * junction_size) / 2
            self.max_x = (position[1] + 1) * cell_size
            self.min_z = position[0] * cell_size + (junction_size / 2)
            self.max_z = position[0] * cell_size + (3 * junction_size) / 2
        elif orientation == "north":
            self.min_x = position[1] * cell_size + (junction_size / 2)
            self.max_x = position[1] * cell_size + (3 * junction_size) / 2
            self.min_z = position[0] * cell_size
            self.max_z = position[0] * cell_size + (junction_size / 2)
        elif orientation == "west":
            self.min_x = position[1] * cell_size
            self.max_x = position[1] * cell_size + junction_size / 2
            self.min_z = position[0] * cell_size + junction_size / 2
            self.max_z = position[0] * cell_size + (3 * junction_size) / 2
        else:
            self.min_x = position[1] * cell_size + (junction_size / 2)
            self.max_x = position[1] * cell_size + (3 * junction_size) / 2
            self.min_z = position[0] * cell_size + (3 * junction_size) / 2
            self.max_z = (position[0] + 1) * cell_size

        outline = np.array([
            [self.max_x, self.max_z],
            [self.max_x, self.min_z],
            [self.min_x, self.min_z],
            [self.min_x, self.max_z]
        ])

        return outline


if __name__ == '__main__':
    c = Corridor(position=(2, 3), orientation="east", cell_size=6, junction_size=3)
    c = Corridor(position=(2, 3), orientation="north", cell_size=6, junction_size=3)
    c = Corridor(position=(2, 3), orientation="west", cell_size=6, junction_size=3)
    c = Corridor(position=(2, 3), orientation="south", cell_size=6, junction_size=3)
