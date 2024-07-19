import math

from matplotlib import pyplot as plt
import numpy as np

from polyapprox.analyses import generate_rand_unif_points, insert_rotations, plot_generator_rotations_against_sum_non_symmetric_coefficients, plot_log_abs_coeffs
from polyapprox.approx import poly_approx
from polyapprox.bases import trig_basis

## 2D Trig ##

def g(theta1, theta2):
    return math.sin(theta1 - theta2) + math.cos(2*(theta1 - theta2)) + math.sin(3*(theta1 - theta2)) + math.sin(4*(theta1 - theta2))

def plot_g_approx_slice(max_degree: int):
    points = generate_rand_unif_points(domains=[(0, 2*np.pi), (0, 2*np.pi)], num_sample_points=300)
    points = insert_rotations(points, 50, [0, 1], "equal-space")
    (approx, coeffs, _) = poly_approx(g, trig_basis, points, max_degree, 2)
    theta_plot = np.linspace(0, 2*np.pi, 1000)
    plt.plot(theta_plot, [g(theta, -theta) for theta in theta_plot], label="f(theta1, -theta1)")
    plt.plot(theta_plot, [approx([theta, -theta]) for theta in theta_plot], label="approx(theta1, -theta1)")
    plt.xlabel("theta1 = -theta2")
    plt.legend()
    plt.show()
    plot_log_abs_coeffs(coeffs, max_degree, 2)

plot_g_approx_slice(13)

plot_generator_rotations_against_sum_non_symmetric_coefficients(
    g,
    domains=[(0, 2*np.pi), (0, 2*np.pi)],
    basis=trig_basis,
    max_degree=13,
    num_sample_points=300,
    generator_rotations=range(100, 200, 4),
    is_coeff_index_symmetric=lambda index: index[0] == index[1] or (index[0] == index[1] + 1 and index[0] % 2 == 0) or (index[0] == index[1] - 1 and index[1] % 2 == 0),
    target_rotation_indices=[0, 1],
    distribution="equal-space"
)