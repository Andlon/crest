import numpy as np


class Mesh2d:
    def __init__(self, vertices, elements):
        # Ensure that the data is contiguous in C-order. This helps interopability with C
        self.__vertices = np.array(vertices, dtype=np.float64, order='C')
        self.__elements = np.array(elements, dtype=np.int32, order='C')
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
