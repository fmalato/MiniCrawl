import numpy as np

from miniworld.miniworld import Room, DEFAULT_WALL_HEIGHT, gen_texcs_floor

from pyglet.gl import (
    GL_POLYGON,
    GL_QUADS,
    glBegin,
    glColor3f,
    glEnd,
    glNormal3f,
    glTexCoord2f,
    glVertex3f,
)


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
        self._contains_stairs = None

    def _render(self):
        """
        Render the static elements of the room
        """

        glColor3f(1, 1, 1)
        self.floor_tex.bind()

        if not self._contains_stairs:
            # Draw the floor
            glBegin(GL_POLYGON)
            glNormal3f(0, 1, 0)
            for i in range(self.floor_verts.shape[0]):
                glTexCoord2f(*self.floor_texcs[i, :])
                glVertex3f(*self.floor_verts[i, :])
            glEnd()
        else:
            for verts, tex_cs in zip(self.floor_verts, self.floor_texcs):
                glBegin(GL_QUADS)
                glNormal3f(0, 1, 0)
                for i in range(verts.shape[0]):
                    glTexCoord2f(*tex_cs[i, :])
                    glVertex3f(*verts[i, :])
                glEnd()

        # Draw the ceiling
        if not self.no_ceiling:
            self.ceil_tex.bind()
            glBegin(GL_POLYGON)
            glNormal3f(0, -1, 0)
            for i in range(self.ceil_verts.shape[0]):
                glTexCoord2f(*self.ceil_texcs[i, :])
                glVertex3f(*self.ceil_verts[i, :])
            glEnd()

        # Draw the walls
        self.wall_tex.bind()
        glBegin(GL_QUADS)
        for i in range(self.wall_verts.shape[0]):
            glNormal3f(*self.wall_norms[i, :])
            glTexCoord2f(*self.wall_texcs[i, :])
            glVertex3f(*self.wall_verts[i, :])
        glEnd()

    def add_portal_on_floor(self, floor_verts):
        """
        Add an opening on the floor
        :param limits: dict containing x-axis and z-axis min/max coord vectors
        """
        self._contains_stairs = True
        self.floor_verts = [
            [
                self.floor_verts[0, :],
                [self.floor_verts[0, 0], 0, floor_verts["lower_right"][2]],
                [self.floor_verts[3, 0], 0, floor_verts["lower_left"][2]],
                self.floor_verts[3, :]
            ],
            [
                [self.floor_verts[0, 0], 0, floor_verts["lower_right"][2]],
                [self.floor_verts[0, 0], 0, floor_verts["upper_right"][2]],
                floor_verts["upper_right"],
                floor_verts["lower_right"]
            ],
            [
                [self.floor_verts[0, 0], 0, floor_verts["upper_right"][2]],
                self.floor_verts[1, :],
                self.floor_verts[2, :],
                [self.floor_verts[2, 0], 0, floor_verts["upper_left"][2]]
            ],
            [
                floor_verts["lower_left"],
                floor_verts["upper_left"],
                [self.floor_verts[3, 0], 0, floor_verts["upper_left"][2]],
                [self.floor_verts[3, 0], 0, floor_verts["lower_left"][2]]
            ]
        ]
        self.floor_verts = np.array(self.floor_verts)
        # Regenerate texture coordinates
        self.floor_texcs = []
        for i in range(self.floor_verts.shape[0]):
            self.floor_texcs.append(gen_texcs_floor(self.floor_tex, self.floor_verts[i]))
        self.floor_texcs = np.array(self.floor_texcs)


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
            self.min_x = position[1] * cell_size + junction_size + (cell_size - junction_size) / 2
            self.max_x = (position[1] + 1) * cell_size
            self.min_z = position[0] * cell_size + (cell_size - junction_size) / 2
            self.max_z = position[0] * cell_size + junction_size + (cell_size - junction_size) / 2
        elif orientation == "north":
            self.min_x = position[1] * cell_size + (cell_size - junction_size) / 2
            self.max_x = position[1] * cell_size + junction_size + (cell_size - junction_size) / 2
            self.min_z = position[0] * cell_size
            self.max_z = position[0] * cell_size + (cell_size - junction_size) / 2
        elif orientation == "west":
            self.min_x = position[1] * cell_size
            self.max_x = position[1] * cell_size + (cell_size - junction_size) / 2
            self.min_z = position[0] * cell_size + (cell_size - junction_size) / 2
            self.max_z = position[0] * cell_size + junction_size + (cell_size - junction_size) / 2
        else:
            self.min_x = position[1] * cell_size + (cell_size - junction_size) / 2
            self.max_x = position[1] * cell_size + junction_size + (cell_size - junction_size) / 2
            self.min_z = position[0] * cell_size + junction_size + (cell_size - junction_size) / 2
            self.max_z = (position[0] + 1) * cell_size

        outline = np.array([
            [self.max_x, self.max_z],
            [self.max_x, self.min_z],
            [self.min_x, self.min_z],
            [self.min_x, self.max_z]
        ])

        return outline
