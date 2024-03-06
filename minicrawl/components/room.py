from miniworld.miniworld import Room, DEFAULT_WALL_HEIGHT


class SquaredRoom(Room):
    def __init__(
            self,
            outline,
            wall_height=DEFAULT_WALL_HEIGHT,
            floor_tex="floor_tiles_bw",
            wall_tex="concrete",
            ceil_text="concrete_tiles",
            no_ceiling=False
    ):
        super().__init__(outline, wall_height, floor_tex, wall_tex, ceil_text, no_ceiling)
