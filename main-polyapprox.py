import math

from matplotlib import pyplot as plt
import numpy as np

from polyapprox.analyses import generate_rand_unif_points, insert_rotations, plot_generator_rotations_against_sum_non_symmetric_coefficients, plot_log_abs_coeffs, symmetry_error
from polyapprox.approx import max_degree_indices, poly_approx
from polyapprox.bases import trig_basis

## 2D Trig ##

def g(theta1, theta2):
    return math.sin(theta1 - theta2) + math.cos(2*(theta1 - theta2)) + math.sin(3*(theta1 - theta2)) + math.sin(4*(theta1 - theta2))

def is_two_trig_coeff_index_symmetric(index: list[int]):
    return index[0] == index[1] or (index[0] == index[1] + 1 and index[0] % 2 == 0) or (index[0] == index[1] - 1 and index[1] % 2 == 0)

import time

start_time = time.time()
def plot_g_approx_slice(max_degree: int):
    points = generate_rand_unif_points(domains=[(0, 2*np.pi), (0, 2*np.pi)], num_sample_points=200)
    points = insert_rotations(points, [0, 1], {"equal_space": 500, "rand_unif": 0})
    (approx, coeffs, _) = poly_approx(g, trig_basis, points, max_degree, 2)
    print(f"{time.time() - start_time} seconds")
    print(symmetry_error(coeffs, max_degree_indices(max_degree, 2), is_two_trig_coeff_index_symmetric))
    theta_plot = np.linspace(0, 2*np.pi, 1000)
    plt.plot(theta_plot, [g(theta, -theta) for theta in theta_plot], label="f(theta1, -theta1)")
    plt.plot(theta_plot, [approx([theta, -theta]) for theta in theta_plot], label="approx(theta1, -theta1)")
    plt.xlabel("theta1 = -theta2")
    plt.legend()
    plt.show()
    plot_log_abs_coeffs(coeffs, max_degree, 2)

plot_g_approx_slice(13)


exit(0)

plot_generator_rotations_against_sum_non_symmetric_coefficients(
    g,
    domains=[(0, 2*np.pi), (0, 2*np.pi)],
    basis=trig_basis,
    max_degree=13,
    num_sample_points=200,
    generator_rotations=range(0, 51, 4),
    is_coeff_index_symmetric=is_two_trig_coeff_index_symmetric,
    target_rotation_indices=[0, 1],
    distributions=["equal-space", "rand-unif"]
)

## 1D Trig ##

def g2(theta):
    return math.cos(6*theta) + math.sin(6*theta) + math.cos(12*theta)

# def plot_1d_trig_rotation_inclusion(max_degree: int):
#     points = generate_rand_unif_points()


## TODO: we need to first prove (or disprove) that the best polynomial approximation, f tilde to f is f tilde sym, the symmetrised f tilde (where symmetrisation is done via Haar integral)
## if we can prove this (most likely can - look for reference online maybe), then find rate of convergence of symmetrised approximation to the best approximation f tilde = f tilde sym, in terms of data that is used to construct a symmetrised approximation (the rate of convergence is determined by ||f tilde - f hat sym||) where f hat sym is symmetric approx we constructed. w.r.t to number of rotations and/or number of initial data poitns
# using least squares, we can (to machine precision) find the best approximation to f, so numerically we can plot the error between best approx and symmetrised approximation
## as check, do approximate integral between f tilde and symmetric approximation, check it is same as L2 norm of non-symmetric coefficients (check the ratio between these two quantities rather than direct comparision)
## try showing this for finite group first (symmetric group or finite rotation group) before continuous group