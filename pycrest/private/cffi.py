import cffi
import os
import numpy as np
from pycrest.mesh import Mesh2d

_THIS_FILE_ABS_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

_ffi = cffi.FFI()
_ffi.cdef("""
typedef struct flat_mesh_data
{
    double * vertices;
    int32_t * elements;
    size_t vertices_size;
    size_t elements_size;
} flat_mesh_data;

void delete_flat_mesh_data(const flat_mesh_data * data);
void delete_flat_mesh_data_array(const flat_mesh_data ** data_array, size_t size);

flat_mesh_data * bisect_to_tolerance(const flat_mesh_data * mesh_data, double tolerance);
flat_mesh_data ** threshold(const flat_mesh_data * initial_mesh,
                           double tolerance,
                           const int32_t * corner_indices,
                           const double * corner_radians,
                           size_t num_corners);
""")
_crest = _ffi.dlopen(_THIS_FILE_ABS_DIR_PATH + "/../../target/release/libpycrest.so")


def _mesh_to_flat_mesh_data(mesh):
    """
    Create an instance of flat_mesh_data from the given mesh for interopability with C.

    The returned instance is only valid as long as the mesh lives, as it does not copy the mesh data.
    :param mesh:
    :return: A CFFI struct of type flat_mesh_data
    """
    data = _ffi.new("struct flat_mesh_data *")
    data.vertices_size = 2 * mesh.num_vertices
    data.elements_size = 3 * mesh.num_elements
    # Here we use the fact that Mesh2d guarantees that the data is C-contiguous
    data.vertices = _ffi.cast("double *", mesh.vertices.ctypes.data)
    data.elements = _ffi.cast("int32_t *", mesh.elements.ctypes.data)
    return data


def _flat_mesh_data_to_mesh(data):
    """
    Construct a Mesh2d instance from the given flat_mesh_data.
    :param data:
    :return:
    """
    num_vertices = int(data.vertices_size / 2)
    num_elements = int(data.elements_size / 3)

    vertex_buffer = _ffi.buffer(data.vertices, data.vertices_size * _ffi.sizeof("double"))
    element_buffer = _ffi.buffer(data.elements, data.elements_size * _ffi.sizeof("int32_t"))

    vertices = np.frombuffer(vertex_buffer, dtype=np.float64).reshape((num_vertices, 2))
    elements = np.frombuffer(element_buffer, dtype=np.int32).reshape((num_elements, 3))
    return Mesh2d(vertices, elements)


def bisect_to_tolerance(initial_mesh, tolerance):
    flat = _mesh_to_flat_mesh_data(initial_mesh)
    flat_result = _crest.bisect_to_tolerance(flat, tolerance)
    mesh_result = _flat_mesh_data_to_mesh(flat_result)
    _crest.delete_flat_mesh_data(flat_result)
    return mesh_result


def threshold(initial_mesh, tolerance, corner_indices, corner_radians):
    corner_indices = np.array(corner_indices, dtype=np.int32, order='C')
    corner_radians = np.array(corner_radians, dtype=np.float64, order='C')

    assert (tolerance > 0)
    assert (len(corner_indices) == len(corner_radians))
    num_corners = len(corner_indices)

    corner_indices_p = _ffi.cast("int32_t *", corner_indices.ctypes.data)
    corner_radians_p = _ffi.cast("double *", corner_radians.ctypes.data)

    flat_initial = _mesh_to_flat_mesh_data(initial_mesh)
    flat_array = _crest.threshold(flat_initial, tolerance, corner_indices_p, corner_radians_p, num_corners)
    coarse_mesh = _flat_mesh_data_to_mesh(flat_array[0])
    fine_mesh = _flat_mesh_data_to_mesh(flat_array[1])
    _crest.delete_flat_mesh_data_array(flat_array, 2)
    return coarse_mesh, fine_mesh
