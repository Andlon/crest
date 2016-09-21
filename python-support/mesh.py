import numpy as np
import json
import os
import subprocess

MESH_PY_ABS_PATH = os.path.dirname(os.path.realpath(__file__))
MESH_BINARY = MESH_PY_ABS_PATH + '/../target/release/mesh'


class Mesh2d:
    def __init__(self, vertices, elements):
        self.__vertices = np.array(vertices)
        self.__elements = np.array(elements)
        self.__vertices.flags.writeable = False
        self.__elements.flags.writeable = False
        assert self.__vertices.ndim is 2
        assert self.__vertices.shape[1] is 2
        assert self.__elements.ndim is 2
        assert self.__elements.shape[1] is 3

    @property
    def vertices(self):
        return self.__vertices

    @property
    def elements(self):
        return self.__elements

    @property
    def num_vertices(self):
        return self.__vertices.shape[0]

    @property
    def num_elements(self):
        return self.__elements.shape[0]


def mesh_from_json(json_string):
    data = json.loads(json_string)
    vertices = np.array(data['vertices'])
    elements = np.array(data['elements'])
    return Mesh2d(vertices, elements)


def refine_mesh(initial_mesh, tolerance, algorithm='bisect'):
    vertices = initial_mesh.vertices.tolist()
    elements = initial_mesh.elements.tolist()
    data = json.dumps({
        'vertices': vertices,
        'elements': elements,
        'tolerance': tolerance,
        'algorithm': algorithm
    })

    result = subprocess.run([MESH_BINARY], input=data, stdout=subprocess.PIPE, check=True, universal_newlines=True)
    return mesh_from_json(result.stdout)

