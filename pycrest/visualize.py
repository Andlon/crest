import matplotlib.pyplot as plt
from matplotlib.pyplot import triplot
from mpl_toolkits.mplot3d import Axes3D

from pycrest.mesh import Mesh2d


def plot_triangulation(tri: Mesh2d, standalone=True, *args, **kwargs):
    figure = plt.figure() if standalone else None
    triplot(tri.vertices[:, 0], tri.vertices[:, 1], tri.elements, *args, **kwargs)
    if figure:
        figure.show()