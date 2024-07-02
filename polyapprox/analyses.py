import math
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from typing import Tuple
from .approx import poly_approx, max_degree_indices

def gen_cheb_points(n):
    return [math.cos(j * np.pi / n) for j in range(n + 1)]

def even_part_with_noise(f):
    return lambda x: (f(x) + f(-x)) / 2 + np.cos(x) * 1e-5 * random.random()

def odd_part(f):
    return lambda x: (f(x) - f(-x)) / 2

NUM_GRID_POINTS = 10000

def analysis(f, domains: list[Tuple[float, float]], bases: list, max_degree: int, num_sample_points: int, perturbation = None):
    dim = len(domains) # number of dimensions
    mesh = np.meshgrid(*[np.linspace(domain[0], domain[1], math.ceil(math.pow(NUM_GRID_POINTS, 1/dim))) for domain in domains])
    grid_points = np.vstack([points.ravel() for points in mesh]).T # equally spaced grid points with the correct dimension
    rand_unif_points = [[random.random() * (domain[1] - domain[0]) + domain[0] for domain in domains] for _ in range(num_sample_points)]
    points_dict = {
        # "cheb": list(itertools.product(*[gen_cheb_points(math.ceil(math.pow(num_sample_points, 1/dim))) for _ in range(dim)])), # Cartesian product of chebyshev points
        # "random_cheb": random.sample(gen_cheb_points(2 * num_sample_points), num_sample_points),
        # "eq_space": np.linspace(-1, 1, num_sample_points).tolist(),
        "random_unif": rand_unif_points,
    }
    if perturbation is not None:
        points_dict["random_unif_rotated"] = [perturbation(x) for x in rand_unif_points]
    prev_X = None
    prev_coeffs = None
    for basis in bases:
        for (points_name, points) in points_dict.items():
            (approx, coeffs, X) = poly_approx(f, basis, points, max_degree, dimension=dim)
            if prev_X is not None:
                diff = np.max(X - prev_X)
                print("max difference in design matrices:", diff)
            if prev_coeffs is not None:
                diff = np.max(np.array(coeffs) - np.array(prev_coeffs))
                print("max difference in coeffs:", diff)
            prev_X = X
            prev_coeffs = coeffs
            max_error_point = grid_points[0]
            max_error = 0 # estimate the max norm error
            for x in grid_points:
                error = np.abs(approx(x) - f(*x))
                if error > max_error:
                    max_error = error
                    max_error_point = x

            print(f"estimated max error (basis: {basis.__name__}, points: {points_name}):", max_error, "at point:", max_error_point.tolist())

            if dim == 1:
                even_coeffs = coeffs[::2]
                odd_coeffs = coeffs[1::2]
                plt.scatter(range(0, max_degree + 1, 2), even_coeffs, marker="o")
                plt.scatter(range(1, max_degree + 1, 2), odd_coeffs, marker="o")
                plt.show()
                plt.plot([x[0] for x in grid_points], [f(x[0]) for x in grid_points], color="blue")
                plt.plot([x[0] for x in grid_points], [approx(x) for x in grid_points], color="red")
                plt.show()
            if dim == 2:
                indices = max_degree_indices(max_degree, dim)
                scatter_plot = plt.scatter([idx[0] for idx in indices], [idx[1] for idx in indices], c=np.log(np.abs(coeffs) + np.finfo(float).eps), cmap="viridis", s=80, alpha=1)
                plt.xlabel("radial basis degree")
                plt.ylabel("angle basis degree")
                plt.colorbar(scatter_plot, label="Log of absolute value of estimated coefficient")
                plt.show()

def max_mean_error(test_fn, epsilon: float, points: list[list[float]], basis, max_degree):
    f = test_fn(epsilon)
    (_, coeffs1, _) = poly_approx(f, basis, points, max_degree, dimension=2)
    perturbation = 2/3 * np.pi
    rotated_points = [[x[0], x[1] + perturbation] for x in points]
    (_, coeffs2, _) = poly_approx(f, basis, rotated_points, max_degree, dimension=2)
    max_error = 0
    error_sum = 0
    assert len(coeffs1) == len(coeffs2)
    for (c1, c2) in zip(coeffs1, coeffs2):
        rel_error = abs(c1 - c2) / max(abs(c1), abs(c2))
        if rel_error > max_error:
            max_error = rel_error 
        error_sum += rel_error
    mean_error = error_sum / len(coeffs1)
    return (max_error, mean_error)

def max_mean_error_analysis(test_fn, max_degree, num_sample_points, basis, powers=range(-10, 11)):
    rho = 1/10000
    random_points = [[random.random() * (domain[1] - domain[0]) + domain[0] for domain in [(rho, 1), (0, 2 * np.pi)]] for _ in range(num_sample_points)]
    epsilon_plot = list(itertools.chain([10**i for i in powers], [5*10**i for i in powers]))
    max_mean_errors = [max_mean_error(test_fn, epsilon, random_points, basis, max_degree) for epsilon in epsilon_plot]
    plt.scatter(np.log10(epsilon_plot), np.log([error[0] for error in max_mean_errors]), s=25, color="red", label="Max error")
    plt.scatter(np.log10(epsilon_plot), np.log([error[1] for error in max_mean_errors]), s=25, color="blue", label="Mean error")
    plt.xlabel("log10(epsilon)")
    plt.ylabel("log(error)")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()