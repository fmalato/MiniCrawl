import numpy as np
import math

from miniworld.entity import Entity, MeshEnt, COLORS
from pyglet.gl import (
    GL_TEXTURE_2D,
    glColor3f,
    glDisable,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glTranslatef,
)

from miniworld.opengl import drawBox


class Stairs(MeshEnt):
    """
    Stairs object to pass from one level to the other
    """

    def __init__(self, color, size=1.5):
        # TODO: absolute path is set to be miniworld/meshes
        super().__init__(height=0.05, mesh_name="stairs_down", static=True)

        if type(size) is int or type(size) is float:
            size = np.array([size, size])
        size = np.array(size)
        sx, sz = size
        sy = 0.05

        self.color = color
        self.size = size

        self.radius = math.sqrt(sx * sx + sz * sz) / 2
        self.height = sy

    def randomize(self, params, rng):
        self.color_vec = COLORS[self.color] + params.sample(rng, "obj_color_bias")
        self.color_vec = np.clip(self.color_vec, 0, 1)

    def render(self):
        """
        Draw the object
        """

        sx, sz = self.size
        sy = 0.05

        glDisable(GL_TEXTURE_2D)
        glColor3f(*self.color_vec)

        glPushMatrix()
        glTranslatef(*self.pos)
        glRotatef(self.dir * (180 / math.pi), 0, 1, 0)

        drawBox(
            x_min=-sx / 2,
            x_max=+sx / 2,
            y_min=0,
            y_max=sy,
            z_min=-sz / 2,
            z_max=+sz / 2,
        )

        glPopMatrix()
