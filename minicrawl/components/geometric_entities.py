import numpy as np
import math

from miniworld.entity import Entity, ObjMesh, COLORS
from pyglet.gl import (
    GL_TEXTURE_2D,
    glColor3f,
    glEnable,
    glDisable,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glTranslatef,
    glScalef
)

from miniworld.opengl import drawBox, Texture
from minicrawl.utils import get_file_path


class MiniCrawlObjMesh(ObjMesh):
    def __init__(self, file_path):
        super().__init__(file_path)

    @classmethod
    def get(self, mesh_name):
        file_path = get_file_path("meshes", mesh_name, "obj")

        if file_path in self.cache:
            return self.cache[file_path]

        mesh = MiniCrawlObjMesh(file_path)
        self.cache[file_path] = mesh

        return mesh


# TODO: works, but going through the floor --> stairs are not rendered
class Stairs(Entity):
    """
    Stairs object to pass from one level to the other.
    It's exactly as a MeshEnt, but with a different path for loading the mesh.
    """
    def __init__(self, height, mesh_name="", static=True):
        super().__init__()

        self.static = static

        # Load the mesh
        self.mesh = MiniCrawlObjMesh.get(mesh_name)

        # Get the mesh extents
        self.sx, self.sy, self.sz = self.mesh.max_coords

        # Compute the mesh scaling factor
        self.scale = 0.5

        # Compute the radius and height (+0.2 for better collision detection)
        self.radius = math.sqrt(self.sx * self.sx + self.sz * self.sz) * self.scale + 0.2
        self.height = height

    def render(self):
        """
        Draw the object
        """
        glPushMatrix()
        glTranslatef(*self.pos)
        glScalef(self.scale, self.scale, self.scale)
        glColor3f(0.5, 0.5, 0.5)
        self.mesh.render()
        glPopMatrix()

    @property
    def is_static(self):
        return self.static


class Stairs2D(Entity):
    """
    Just like a Box, but much flatter and with texture
    """
    def __init__(self, color, tex_name="", edge_size=0.8):
        super().__init__()

        if type(edge_size) is int or type(edge_size) is float:
            edge_size = np.array([edge_size, edge_size])
        edge_size = np.array(edge_size)
        sx, sz = edge_size
        sy = 0.05

        self.color = color
        self.size = np.array([sx, sy, sz])

        self.radius = math.sqrt(sx * sx + sz * sz) / 2
        self.height = sy

        self.tex_name = tex_name
        self.texture = Texture.get(tex_name)

    def randomize(self, params, rng):
        self.color_vec = COLORS[self.color] + params.sample(rng, "obj_color_bias")
        self.color_vec = np.clip(self.color_vec, 0, 1)

    def render(self):
        """
        Draw the object
        """

        sx, sy, sz = self.size

        glEnable(GL_TEXTURE_2D)
        self.texture.bind()
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
