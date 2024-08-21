import math
from matplotlib import pyplot as plt
import numpy as np
from bases import cheb_basis
from cusps.approx import points_approx
from scipy.spatial import Delaunay

from cusps.transition import barycentric_coords
from polyapprox.analyses import plotting_grid_points

def simplex_area(simplex_vertices):
    x1, y1 = simplex_vertices[0]
    x2, y2 = simplex_vertices[1]
    x3, y3 = simplex_vertices[2]

    area = 0.5 * np.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    return area

def are_coords_in_simplex(coords):
    return all(0 <= c and c <= 1 for c in coords)

def filter_points_in_simplex(points, simplex_vertices):
    tri = Delaunay(simplex_vertices)
    barycentre = np.sum(simplex_vertices, axis=0)/len(simplex_vertices)
    assert len(barycentre) == 2
    return list(filter(lambda p:
        (tri.find_simplex(p) == 0)
        # and all(np.linalg.norm(p - v) >= 0.1 for v in simplex_vertices)
        , points))

def plot_error(f, num_sample_points: int, simplex_vertices, max_degree, num_test_points: int):
    area_ratio = simplex_area(simplex_vertices) / 4
    sample_points = np.random.uniform(-1, 1, (math.ceil(num_sample_points / area_ratio), 2))
    sample_points = plotting_grid_points([(-1, 1), (-1, 1)], math.ceil(num_sample_points / area_ratio))
    sample_points = filter_points_in_simplex(sample_points, simplex_vertices)
    print(len(sample_points))
    approx, coeffs = points_approx(cheb_basis, np.array(sample_points), np.array([f(*point) for point in sample_points]), max_degree)
    test_points = np.random.uniform(-1, 1, (math.ceil(num_test_points / area_ratio), 2))
    test_points = plotting_grid_points([(-1, 1), (-1, 1)], math.ceil(num_test_points / area_ratio))
    test_points = filter_points_in_simplex(test_points, simplex_vertices)
    print(len(test_points))
    errors = [abs(f(*point) - approx(point)) for point in test_points]
    plt.scatter(*zip(*test_points), c=np.log10(errors), cmap="viridis")
    plt.clim(0, -8)
    plt.title("log10 error")
    plt.colorbar()
    plt.show()

def gen_sinh_func(a: float, b: float):
    return lambda x, y: math.sinh(math.sqrt(a**2 * y**2 + b**2 * x**2)/(a*b))

plot_error(gen_sinh_func(0.2, 0.2), 100, np.array([[0, 0], [0.3, 0.6], [0, 0.6]]), 20, 4000)