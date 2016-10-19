from pycrest.mesh import Mesh2d
from pycrest.pycrest_cffi import _mesh_to_flat_mesh_data, _flat_mesh_data_to_mesh
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal


def test_mesh_flat_data_roundtrip():
    vertices = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0)
    ]
    elements = [
        (0, 1, 3),
        (1, 2, 3)
    ]
    mesh = Mesh2d(vertices, elements)

    flat = _mesh_to_flat_mesh_data(mesh)
    converted_mesh = _flat_mesh_data_to_mesh(flat)

    assert_array_almost_equal(mesh.vertices, converted_mesh.vertices)
    assert_array_equal(mesh.elements, converted_mesh.elements)

