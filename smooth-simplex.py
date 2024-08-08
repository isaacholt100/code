import math
from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay

from cusps.transition import simplex_approx_weight

# def barycentric_coords(x, y, x1, y1, x2, y2, x3, y3):
#     det = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
#     lambda1 = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / det
#     lambda2 = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / det
#     lambda3 = 1 - lambda1 - lambda2
#     return lambda1, lambda2, lambda3

# def f(x):
#     return np.floor((np.sign(x) + 1)/2) * np.exp(-1/x)

# def g(x):
#     return f(x) / (f(x) + f(1 - x))

# def smooth_step(x):
#     a = -0.1
#     b = 0
#     return g((x - a)/(b - a))

# def b(x):
#     a = 0
#     b = 1
#     return g((x - a)/(b - a))

# def triangle_transition(x, y, x1, y1, x2, y2, x3, y3):
#     lambda1, lambda2, lambda3 = barycentric_coords(x, y, x1, y1, x2, y2, x3, y3)
#     return smooth_step(lambda1) * smooth_step(lambda2) * smooth_step(lambda3) * g(np.linalg.norm(np.array([x - x1, y - y1]))**2) * g(np.linalg.norm(np.array([x - x2, y - y2]))**2) * g(np.linalg.norm(np.array([x - x3, y - y3]))**2)

# x_plot, y_plot = np.meshgrid(np.linspace(-1, 2, 1000), np.linspace(-1, 2, 1000), indexing="ij")
# z_plot = triangle_transition(x_plot, y_plot, 0, 0, 1, 0, 0, 1.5)

def barycentric_coords(x, x1, x2, x3):
    det = (x2[1] - x3[1])*(x1[0] - x3[0]) + (x3[0] - x2[0])*(x1[1] - x3[1])
    lambda1 = ((x2[1] - x3[1])*(x[0] - x3[0]) + (x3[0] - x2[0])*(x[1] - x3[1])) / det
    lambda2 = ((x3[1] - x1[1])*(x[0] - x3[0]) + (x1[0] - x3[0])*(x[1] - x3[1])) / det
    lambda3 = 1 - lambda1 - lambda2
    return [lambda1, lambda2, lambda3]

def f(x):
    if x > 0:
        return np.exp(-1/x)
    return 0

def g(x):
    return f(x) / (f(x) + f(1 - x))

def b(x):
    a = 0
    b = 0.1
    return g((x - a)/(b - a))

def smooth_step(a, b, x):
    return g((x - a)/(b - a))

def triangle_transition(x, x1, x2, x3):
    import math
    return math.prod(smooth_step(0, 0.1, l) for l in barycentric_coords(x, x1, x2, x3)) * b(np.linalg.norm(x - x1)**2) * b(np.linalg.norm(x - x2)**2) * b(np.linalg.norm(x - x3)**2)

def _barycentric_coords_new(triangulation: Delaunay, simplex_index: int, x):
    b = triangulation.transform[simplex_index, :2].dot(np.transpose(x - triangulation.transform[simplex_index, 2]))
    return b.tolist() + [1 - b.sum(axis=0)]


def simplex_approx_weight_1(triangulation: Delaunay, x, simplex_index: int) -> float:
    return math.prod(smooth_step(0, 0.3, l) for l in _barycentric_coords_new(triangulation, simplex_index, x))

triangulation = Delaunay(np.array([[1, 1], [0, 1], [0, 0]]))
print(triangulation.simplices)
print(_barycentric_coords_new(triangulation, 0, [0.2, 0.5]))

x_plot, y_plot = np.meshgrid(np.linspace(-0.1, 1.1, 300), np.linspace(-0.1, 1.1, 300), indexing="ij")
input_values = np.vstack((x_plot.flatten(), y_plot.flatten())).T
z_plot = [simplex_approx_weight(0.2, 0.4, [[1, 1], [0, 1], [0, 0]], triangulation, x, 0) for x in input_values]
z_plot = np.array(z_plot).reshape(x_plot.shape)

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.contour3D(x_plot, y_plot, z_plot, 100, cmap="binary")
# # ax.view_init(30, 35)
# plt.show()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# The pyvista plot starts here
pl = pv.Plotter()
scalefactor = np.abs(np.max(z_plot)) # make axes look like similar size to avoid strange shape
mesh1 = pv.StructuredGrid(x_plot, y_plot, z_plot)

pl.add_mesh(mesh1,opacity=1,color=colors[0],show_edges=False,edge_color=colors[0],point_size=15)

pl.show_grid(axes_ranges=[-2, 2, -2, 2,np.min(z_plot), np.max(z_plot)],ztitle='z', xtitle='x', ytitle='y',grid='back', all_edges=False,padding=0.1,show_zlabels=True,n_xlabels=5,n_ylabels=5,n_zlabels=4,use_3d_text=False)
pl.show(auto_close=False)

# If you need to save a figure, you have to manually adjust the camera angle with features like these.
# You will also have to add something like font_size=70 to the pl.show_grid function keywords since the font will appear too small on the saved figure.
# Uncomment and play with the following to save a figure (good luck!)
#pl.camera.position = (5.0,6.0,6.0)
#pl.camera.elevation = -20
#pl.camera.azimuth -= 100
#pl.window_size = 3000, 3000
pl.save_graphic("pyvista_plot.pdf")

pl.close()
