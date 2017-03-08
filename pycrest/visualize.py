import matplotlib.pyplot as plt
from matplotlib.pyplot import triplot
from mpl_toolkits.mplot3d import Axes3D

from pycrest.mesh import Mesh2d


def plot_triangulation(tri: Mesh2d, standalone=True, *args, **kwargs):
    figure = plt.figure() if standalone else None
    triplot(tri.vertices[:, 0], tri.vertices[:, 1], tri.elements, *args, **kwargs)
    xmin = tri.vertices[:, 0].min()
    xmax = tri.vertices[:, 0].max()
    ymin = tri.vertices[:, 1].min()
    ymax = tri.vertices[:, 1].max()
    padding = 0.05 * (max(xmax - xmin, ymax - ymin))
    plt.axis('square')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((xmin - padding, xmax + padding))
    plt.ylim((ymin - padding, ymax + padding))
    if figure:
        figure.show()
