import math
from typing import Callable, Iterable

from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import quad
import itertools

from polyapprox.analyses import estimate_max_abs_error, filter_symmetrise, generate_rand_unif_points, insert_permutations, insert_rotations, plot_generator_rotations_against_sum_non_symmetric_coefficients, plot_log_abs_coeffs, symmetrise, symmetry_error, l2_distance
from polyapprox.approx import max_degree_indices, poly_approx
from bases import cheb_basis, exponential_basis, trig_basis, trig_basis_l2_normalised

## 2D Trig ##

## TODO: try with different intervals [a, b] other than [-pi, pi] or [0, 2*pi]

def g(theta1: float, theta2: float) -> float:
    return math.sin(theta1 - theta2) + math.cos(2*(theta1 - theta2)) + math.sin(3*(theta1 - theta2)) + math.sin(5*(theta1 - theta2))

def trig_complex_1(theta1: float, theta2: float) -> float:
    return np.exp(1j*(theta1 - theta2)) + np.exp(2j*(theta1 - theta2)) + np.exp(3j*(theta2 - theta1)) - np.exp(20j*(theta2 - theta1))

def trig_complex_2(theta1: float, theta2: float) -> float:
    return (np.exp(7j*(theta1 - theta2)) + np.exp(5j*(theta1 - theta2))) / (1 + np.exp(1j*(theta1 - theta2))/2)

def trig_complex_3(theta1: float, theta2: float) -> float:
    return (theta1 - theta2) % (2*np.pi)

import time

def is_exp_basis_coeff_index_symmetric(idx: list[int]) -> bool:
    return (idx[0] == 0 and idx[1] == 0) or (idx[0] != idx[1] and (idx[0] + 1)//2 == (idx[1] + 1)//2)

start_time = time.time()
def two_d_trig(f: Callable[[float, float], float], max_degree: int):
    import scipy.stats
    # play around with deplting some of the points
    # try to prove that the equispaced points gives convergence
    points = generate_rand_unif_points(domains=[(0, 2*np.pi), (0, 2*np.pi)], num_sample_points=500)

    points, rr = insert_rotations(points, [0, 1], {"equal_space": 20, "rand_unif": 0, "cheb": 0}, inclusion_rate=1)
    
    (approx, coeffs) = poly_approx(f, exponential_basis, points, max_degree, 2)
    # print(np.abs(coeffs))
    # xy = np.vstack([*zip(*points)])
    # kde = scipy.stats.gaussian_kde(xy)

    # xgrid = np.linspace(0, 2*np.pi, 100)
    # ygrid = np.linspace(0, 2*np.pi, 100)
    # Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    # Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()])).reshape(Xgrid.shape)
    # plt.imshow(Z, origin='lower', aspect='auto', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='viridis')
    # plt.colorbar()
    # plt.show()

    # theta_plot = np.linspace(0, 2*np.pi, 1000)
    # plt.plot(theta_plot, [g(theta, -theta) for theta in theta_plot], label="f(theta1, -theta1)")
    # plt.plot(theta_plot, [approx(theta, -theta) for theta in theta_plot], label="approx(theta1, -theta1)")
    # plt.xlabel("theta1 = -theta2")
    # plt.legend()
    # plt.show()

    # plt.plot(theta_plot, [g(theta, 0) for theta in theta_plot], label="f(theta1, -theta1)")
    # plt.plot(theta_plot, [approx(theta, 0) for theta in theta_plot], label="approx(theta1, -theta1)")
    # plt.xlabel("theta1 = -theta2")
    # plt.legend()
    # plt.show()
    # print(f"{time.time() - start_time} seconds")
    print("L2 symmetry error:", symmetry_error(coeffs, max_degree_indices(max_degree, 2), is_exp_basis_coeff_index_symmetric))

    # print(rr)
    # print([approx(2 + 2*j*np.pi/40, 1 + 2*j*np.pi/40) for j in range(40)])
    print(approx(2))

    indices = list(filter(lambda idx: is_exp_basis_coeff_index_symmetric(idx), max_degree_indices(max_degree, 2)))
    # print(max_degree_indices(max_degree, 2))
    # approx_sym = filter_symmetrise(coeffs, indices, is_exp_basis_coeff_index_symmetric, exponential_basis)
    # print("l2 sym error via integral:", l2_distance(approx_sym, approx, domains=[(0, 2*np.pi), (0, 2*np.pi)]))

    plt.scatter(*zip(*indices))
    plt.show()

    # print("L2 approximation error:", l2_distance(approx, f, domains=[(0, 2*np.pi), (0, 2*np.pi)]))
    plot_log_abs_coeffs(coeffs, max_degree, 2)

two_d_trig(trig_complex_2, 4)

def plot_sym_error_against_inclusion_probability(f, max_degree: int, probabilities: Iterable[float]):
    init_points = generate_rand_unif_points(domains=[(0, 2*np.pi), (0, 2*np.pi)], num_sample_points=500)
    sym_errors = []
    for p in probabilities:
        print("progress:", p)
        points = insert_rotations(init_points, [0, 1], {"equal_space": 8, "rand_unif": 0, "cheb": 0}, inclusion_rate=p)
        (approx, coeffs) = poly_approx(f, exponential_basis, points, max_degree, 2)
        sym_errors.append(symmetry_error(coeffs, max_degree_indices(max_degree, 2), is_exp_basis_coeff_index_symmetric))
    plt.plot(list(probabilities), np.log10(sym_errors))
    plt.show()

# plot_sym_error_against_inclusion_probability(trig_complex_2, 12, np.linspace(0, 0.999, 30))

## TODO: check whether coeffs are same after convergence (for eq space points)

def plot_rand_vs_eq_space_rotations_2d_trig(f, max_degree: int, num_rotations_range: Iterable[int]):
    init_points = generate_rand_unif_points(domains=[(0, 2*np.pi), (0, 2*np.pi)], num_sample_points=500)
    eq_space_sym_errors = []
    rand_unif_sym_errors = []
    indices = max_degree_indices(max_degree, 2)
    for num_rotations in num_rotations_range:
        print("progress:", num_rotations)
        points = insert_rotations(init_points, [0, 1], {"equal_space": num_rotations, "rand_unif": 0, "cheb": 0})
        (approx, coeffs) = poly_approx(f, exponential_basis, points, max_degree, 2)
        sym_error = symmetry_error(coeffs, indices, is_exp_basis_coeff_index_symmetric)
        eq_space_sym_errors.append(sym_error)

        points = insert_rotations(init_points, [0, 1], {"equal_space": 1, "rand_unif": num_rotations - 1, "cheb": 0})
        (approx, coeffs) = poly_approx(f, exponential_basis, points, max_degree, 2)
        sym_error = symmetry_error(coeffs, indices, is_exp_basis_coeff_index_symmetric)
        rand_unif_sym_errors.append(sym_error)
    
    plt.plot(list(num_rotations_range), np.log10(eq_space_sym_errors), label="eq space")
    plt.plot(list(num_rotations_range), np.log10(rand_unif_sym_errors), label="rand unif")
    plt.legend()
    plt.xlabel("Number of rotations")
    plt.ylabel("log10 of symmetry error")
    plt.show()

# plot_rand_vs_eq_space_rotations_2d_trig(trig_complex_2, 14, range(80, 200, 4))

N = 3

def g2(theta):
    return math.cos(N*theta) + math.sin(N*theta) + math.cos(2*N*theta) - 2*math.sin(2*N*theta)

def is_one_trig_n_point_coeff_index_symmetric(n: int):
    return lambda index: ((index[0] + 1) // 2) % n == 0

def one_d_trig(max_degree: int):
    points = generate_rand_unif_points(domains=[(0, 2*np.pi)], num_sample_points=16)
    points = insert_rotations(points, [0], {"equal_space": N, "rand_unif": 0, "cheb": 0})
    (approx, coeffs) = poly_approx(g2, trig_basis, points, max_degree, 1)
    indices = max_degree_indices(max_degree, 1)
    print([approx(2.2 + 2*j*np.pi/N) for j in range(N)])
    # sym_error = symmetry_error(coeffs, max_degree_indices(max_degree, 1), is_one_trig_n_point_coeff_index_symmetric(N))
    # print("symmetry error:", sym_error)
    # g2_sym_a = filter_symmetrise(coeffs, indices, is_one_trig_n_point_coeff_index_symmetric(N), trig_basis)
    # print("ratio a:", sym_error / l2_distance(approx, g2_sym_a, [(0, 2*np.pi)]))
    # def ga(g, args):
    #     return [g * 2*np.pi/N + args[0]]
    # g2_sym_b = symmetrise(approx, list(range(N)), ga)
    # print("ratio b:", sym_error / l2_distance(approx, g2_sym_b, [(0, 2*np.pi)]))
    # assert estimate_max_abs_error(g2_sym_a, g2_sym_b, list(map(lambda x: [x], np.linspace(0, 2*np.pi, 1000))))[0] <= 1e-14
    # plot_log_abs_coeffs(coeffs, max_degree, 1)



# one_d_trig(7)

def n_d_permutation_invariant(n_variables: int, f, max_degree: int):
    domains = [(-1.0, 1.0) for _ in range(n_variables)]
    points = generate_rand_unif_points(domains=domains, num_sample_points=20)
    print(len(points))
    points = insert_permutations(points)
    print(len(points))
    (approx, coeffs) = poly_approx(f, cheb_basis, points, max_degree, n_variables)
    print(approx)
    def ga(g, args):
        return [args[i] for i in g]
    approx_sym = symmetrise(approx, list(itertools.permutations(range(n_variables))), ga)
    sym_error = l2_distance(approx, approx_sym, domains)
    print(sym_error)

def sym_3(x1, x2, x3):
    return math.cos(x1*x2*x3) + math.sin(x1+x2+x3) * math.cos(x1*x2 + x1*x3 + x2*x3) + math.cos(x1+x2+x3)**5 * math.sin(x1*x2 + x1*x3 + x2*x3)**7 + (math.cos(x1*x2 + x1*x3 + x2*x3) + math.cos(x1+x2+x3)**5 * math.sin(x1*x2 + x1*x3 + x2*x3)**7)**9

# n_d_permutation_invariant(3, sym_3, 6)

exit(0)

plot_generator_rotations_against_sum_non_symmetric_coefficients(
    f,
    domains=[(0, 2*np.pi), (0, 2*np.pi)],
    basis=trig_basis,
    max_degree=13,
    num_sample_points=200,
    generator_rotations=range(0, 51, 4),
    is_coeff_index_symmetric=is_two_trig_coeff_index_symmetric,
    target_rotation_indices=[0, 1],
    distributions=["equal-space", "rand-unif"]
)
