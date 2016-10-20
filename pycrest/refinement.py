from .private import cffi


def bisect_to_tolerance(initial_mesh, tolerance):
    return cffi.bisect_to_tolerance(initial_mesh, tolerance)


def threshold(initial_mesh, tolerance, corner_indices, corner_radians):
    return cffi.threshold(initial_mesh, tolerance, corner_indices, corner_radians)
