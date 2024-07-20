import math
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from typing import Iterable, Literal, Tuple, TypedDict
from polyapprox.approx import poly_approx, max_degree_indices

NUM_GRID_POINTS = 10000

np.random.seed(42)

# return list with rotated copies of each point inserted, in addition to the original points
class RotationInsertions(TypedDict):
    equal_space: int
    rand_unif: int
def insert_rotations(points, target_indices: list[int], insertions: RotationInsertions):
    for i in range(len(points)): # use index loop rather than list item loop as we are modifying the list in the loop
        point = points[i]
        for j in range(insertions["equal_space"]):
            rotation = j/insertions["equal_space"] * 2 * np.pi
            new_point = point.copy()
            for k in target_indices:
                new_point[k] += rotation
            points.append(new_point)
        for j in range(insertions["rand_unif"]):
            rotation = random.random() * 2 * np.pi
            new_point = point.copy()
            for k in target_indices:
                new_point[k] += rotation
            points.append(new_point)
    return points

def generate_rand_unif_points(domains: list[tuple[float, float]], num_sample_points: int):
    return [[np.random.random() * (domain[1] - domain[0]) + domain[0] for domain in domains] for _ in range(num_sample_points)]

def estimate_max_abs_error(f, approx, sample_points):
    max_error_point = sample_points[0]
    max_error = 0 # estimate the max norm error
    for x in sample_points:
        error = np.abs(approx(x) - f(*x))
        if error > max_error:
            max_error = error
            max_error_point = x
    return max_error, max_error_point

def estimate_max_abs_relative_error(f, approx, sample_points):
    max_error_point = sample_points[0]
    max_error = 0 # estimate the max norm error
    for x in sample_points:
        fx = f(*x)
        error = np.abs(approx(x) - fx)/fx
        if error > max_error:
            max_error = error
            max_error_point = x
    return max_error, max_error_point

def plotting_grid_points(domains: list[Tuple[float, float]], num_grid_points: int = NUM_GRID_POINTS):
    dim = len(domains)
    mesh = np.meshgrid(*[np.linspace(domain[0], domain[1], math.ceil(math.pow(num_grid_points, 1/dim))) for domain in domains])
    return np.vstack([points.ravel() for points in mesh]).T

def plot_log_abs_coeffs(coeffs, max_degree: int, dim: int):
    assert dim == 1 or dim == 2
    indices = max_degree_indices(max_degree, dim)
    y_scatter = [0] * len(indices) if dim == 1 else [idx[1] for idx in indices]
    scatter_plot = plt.scatter([idx[0] for idx in indices], y_scatter, c=np.log10(np.abs(coeffs) + np.finfo(float).eps), cmap="viridis", s=80, alpha=1)
    plt.xlabel("basis degree")
    plt.colorbar(scatter_plot, label="Log of absolute value of estimated coefficient")
    plt.show()

def coefficient_convergence_analysis(f, num_coeffs, basis, num_sample_points, domains, degrees):
    coefficients = []
    rand_unif_points = [[random.random() * (domain[1] - domain[0]) + domain[0] for domain in domains] for _ in range(num_sample_points)] 
    for max_degree in degrees:
        (approx, coeffs, X) = poly_approx(f, basis, rand_unif_points, max_degree, dimension=len(domains))
        coefficients.append(coeffs[0:num_coeffs])
    coefficients = np.transpose(coefficients)
    for i in range(coefficients.shape[0]):
        plt.scatter(list(degrees)[0:-1], np.log10(np.abs(coefficients[i, 1:] - coefficients[i, 0:-1])), label=f"coefficient {i}")
    plt.xlabel("Approximation degree")
    plt.ylabel("Log10 of absolute value of coefficient change between current and next degree")
    plt.legend()
    plt.show()

# NOTE: only works for filter cases (since computes the 2-norm of the non-symmetric coeffs)
def symmetry_error(coeffs, indices: list[list[int]], is_coeff_index_symmetric) -> float:
    non_symmetric_coeffs_2_norm = 0
    for (i, index) in enumerate(indices):
        if not is_coeff_index_symmetric(index):
            non_symmetric_coeffs_2_norm += np.abs(coeffs[i])**2
    return np.sqrt(non_symmetric_coeffs_2_norm)

def plot_generator_rotations_against_sum_non_symmetric_coefficients(f, domains: list[Tuple[float, float]], basis, max_degree: int, num_sample_points: int, generator_rotations: Iterable[int], is_coeff_index_symmetric, target_rotation_indices, distributions: list[Literal["equal-space", "rand-unif"]]):
    dim = len(domains) # number of dimensions
    print(np.random.random())
    init_rand_unif_points = generate_rand_unif_points(domains, num_sample_points)
    errors = []
    indices = max_degree_indices(max_degree, dim)
    non_symmetric_coeff_indices = list(filter(lambda idx: not is_coeff_index_symmetric(idx), indices))
    for n in generator_rotations:
        rand_unif_points = init_rand_unif_points.copy()
        print("progress:", n)
        rand_unif_points = insert_rotations(rand_unif_points, target_rotation_indices, {"equal_space": n, "rand_unif": 0})
        (approx, coeffs, X) = poly_approx(f, basis, rand_unif_points, max_degree, dimension=dim)
        errors.append(symmetry_error(coeffs, indices, is_coeff_index_symmetric))
    plt.scatter(list(generator_rotations), np.log10(np.array(errors)))
    plt.ylabel("log10 of 2-norm of non-symmetric coefficients")
    plt.xlabel(f"number of extra rotations ({distributions})")
    plt.show()

def depletion_probability_analysis(f, domains: list[Tuple[float, float]], basis, max_degree: int, num_sample_points: int, inclusion_probabilities, is_coeff_index_symmetric):
    dim = len(domains) # number of dimensions
    init_rand_unif_points = generate_rand_unif_points(domains, num_sample_points)
    errors = []
    for p in inclusion_probabilities:
        rand_unif_points = init_rand_unif_points.copy()
        print("progress:", p)
        for i in range(len(rand_unif_points)):
            if random.random() < p:
                point = rand_unif_points[i]
                rand_unif_points.append([point[0], point[1] + 2/3*np.pi])
                rand_unif_points.append([point[0], point[1] + 4/3*np.pi])
        (approx, coeffs, X) = poly_approx(f, basis, rand_unif_points, max_degree, dimension=dim)
        indices = max_degree_indices(max_degree, dim)
        errors.append(symmetry_error(coeffs, indices, is_coeff_index_symmetric))
    plt.ylabel("log10 of 2-norm of non-symmetric coefficients")
    plt.xlabel(f"probability of including extra rotations")
    plt.scatter(inclusion_probabilities, np.log10(errors))
    plt.show()

def symmetrise(f, group: list, group_action):
    def f_sym(x):
        return 1/len(group)*sum(f(group_action(g, x)) for g in group)
    return f_sym