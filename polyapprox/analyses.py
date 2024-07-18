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
    for i in range(len(rand_unif_points)):
        # if random.random() < 0.99:
            point = rand_unif_points[i]
            n = 50
            for i in range(n):
                r1 = i/n * 2 * np.pi
                rand_unif_points.append([point[0] + r1, point[1] + r1])
                rand_unif_points.append([point[0] + r1, point[1] + r1])
            # for i in range(10):
            #     r1 = random.random() * 2 * np.pi
            #     rand_unif_points.append([point[0] + r1, point[1] + r1])
    points_dict = {
        # "cheb": list(itertools.product(*[gen_cheb_points(math.ceil(math.pow(num_sample_points, 1/dim))) for _ in range(dim)])), # Cartesian product of chebyshev points
        # "random_cheb": random.sample(gen_cheb_points(2 * num_sample_points), num_sample_points),
        # "eq_space": np.linspace(-1, 1, num_sample_points).tolist(),
        "random_unif": rand_unif_points,
    }
    if perturbation is not None:
        points_dict["random_unif_perturbed"] = [perturbation(x) for x in rand_unif_points]
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
                indices = max_degree_indices(max_degree, dim)
                scatter_plot = plt.scatter([idx[0] for idx in indices], [0] * len(indices), c=np.log10(np.abs(coeffs) + np.finfo(float).eps), cmap="viridis", s=80, alpha=1)
                plt.xlabel("basis degree")
                plt.colorbar(scatter_plot, label="Log of absolute value of estimated coefficient")
                plt.show()
            if dim == 2:
                indices = max_degree_indices(max_degree, dim)
                scatter_plot = plt.scatter([idx[0] for idx in indices], [idx[1] for idx in indices], c=np.log10(np.abs(coeffs) + np.finfo(float).eps), cmap="viridis", s=80, alpha=1)
                # plt.xlabel("radial basis degree")
                # plt.ylabel("angle basis degree")
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

def generator_rotations_analysis(f, domains: list[Tuple[float, float]], basis, max_degree: int, num_sample_points: int, generator_rotations: list):
    dim = len(domains) # number of dimensions
    mesh = np.meshgrid(*[np.linspace(domain[0], domain[1], math.ceil(math.pow(NUM_GRID_POINTS, 1/dim))) for domain in domains])
    grid_points = np.vstack([points.ravel() for points in mesh]).T # equally spaced grid points with the correct dimension
    init_rand_unif_points = [[random.random() * (domain[1] - domain[0]) + domain[0] for domain in domains] for _ in range(num_sample_points)]
    errors = []
    for n in generator_rotations:
        rand_unif_points = init_rand_unif_points.copy()
        print("progress:", n)
        for i in range(len(rand_unif_points)):
            if random.random() < 1:
                point = rand_unif_points[i]
                for i in range(n):
                    r1 = i/n * 2 * np.pi
                    rand_unif_points.append([point[0] + r1, point[1] + r1])
        (approx, coeffs, X) = poly_approx(f, basis, rand_unif_points, max_degree, dimension=dim)
        max_error_point = grid_points[0]
        max_error = 0 # estimate the max norm error
        for x in grid_points:
            error = np.abs(approx(x) - f(*x))
            if error > max_error:
                max_error = error
                max_error_point = x
        errors.append(max_error)
    plt.scatter(generator_rotations, np.log10(errors))
    plt.show()

def depletion_probability_analysis(f, domains: list[Tuple[float, float]], basis, max_degree: int, num_sample_points: int, inclusion_probabilities):
    dim = len(domains) # number of dimensions
    mesh = np.meshgrid(*[np.linspace(domain[0], domain[1], math.ceil(math.pow(NUM_GRID_POINTS, 1/dim))) for domain in domains])
    grid_points = np.vstack([points.ravel() for points in mesh]).T # equally spaced grid points with the correct dimension
    init_rand_unif_points = [[random.random() * (domain[1] - domain[0]) + domain[0] for domain in domains] for _ in range(num_sample_points)]
    grid = np.meshgrid(*[np.linspace(domain[0], domain[1], math.ceil(math.sqrt(num_sample_points))) for domain in domains])
    init_rand_unif_points = np.vstack([coord.ravel() for coord in grid]).T.tolist()
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
        max_error_point = grid_points[0]
        max_error = 0 # estimate the max norm error
        for x in grid_points:
            error = np.abs(approx(x) - f(*x))
            if error > max_error:
                max_error = error
                max_error_point = x
        errors.append(max_error)
    plt.scatter(inclusion_probabilities, errors)
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
    plt.scatter(np.log10(epsilon_plot), np.log10([error[0] for error in max_mean_errors]), s=25, color="red", label="Max error")
    plt.scatter(np.log10(epsilon_plot), np.log10([error[1] for error in max_mean_errors]), s=25, color="blue", label="Mean error")
    plt.xlabel("log10(epsilon)")
    plt.ylabel("log(error)")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()