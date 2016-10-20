from .private import cffi


def bisect_to_tolerance(initial_mesh, tolerance):
    return cffi.bisect_to_tolerance(initial_mesh, tolerance)
