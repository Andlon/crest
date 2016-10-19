import cffi
import os
import numpy as np
from .mesh import Mesh2d

_THIS_FILE_ABS_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

_ffi = cffi.FFI()
_ffi.cdef("""
typedef struct flat_mesh_data
{
    double * vertices;
    int * elements;
    size_t vertices_size;
    size_t elements_size;
} flat_mesh_data;

void delete_flat_mesh_data(const flat_mesh_data * data);
flat_mesh_data * bisect_to_tolerance(const flat_mesh_data * mesh_data, double tolerance);
""")
_crest = _ffi.dlopen(_THIS_FILE_ABS_DIR_PATH + "/../target/release/libpycrest.so")


def _mesh_to_flat_mesh_data(mesh):
    vertices = np.ascontiguousarray(mesh.vertices, np.float64)
    elements = np.ascontiguousarray(mesh.elements, np.int32)

    # TODO: Fix the below mess! The below is not at all safe, because if vertices and elements are actually copies,
    # they fall out of scope after the ffi cast, and in which case their data is no longer valid.
    data = _ffi.new("struct flat_mesh_data *")
    data.vertices_size = 2 * mesh.num_vertices
    data.elements_size = 3 * mesh.num_elements
    data.vertices = _ffi.cast("double *", vertices.ctypes.data)
    data.elements = _ffi.cast("int *", elements.ctypes.data)
    return data


def _flat_mesh_data_to_mesh(data):
    num_vertices = int(data.vertices_size / 2)
    num_elements = int(data.elements_size / 3)

    vertex_buffer = _ffi.buffer(data.vertices, data.vertices_size * _ffi.sizeof("double"))
    element_buffer = _ffi.buffer(data.elements, data.elements_size * _ffi.sizeof("int"))

    vertices = np.frombuffer(vertex_buffer, dtype=np.float64).reshape((num_vertices, 2))
    elements = np.frombuffer(element_buffer, dtype=np.int32).reshape((num_elements, 3))
    return Mesh2d(vertices, elements)


def bisect_to_tolerance(initial_mesh, tolerance):
    flat = _mesh_to_flat_mesh_data(initial_mesh)
    flat_result = _crest.bisect_to_tolerance(flat, tolerance)
    print("Finished bisection!")
    mesh_result = _flat_mesh_data_to_mesh(flat_result)
    _crest.delete_flat_mesh_data(flat_result)
    return mesh_result