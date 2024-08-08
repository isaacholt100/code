import numpy as np
from cusps.approx import points_approx


def plot_error(f, matched_cusp_point, num_sample_points: int, simplex_vertices, max_degree):
    sample_points = []
    for _ in range(num_sample_points):
        
    approx = points_approx(cheb_basis, max_degree)