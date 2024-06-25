import math
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from typing import Tuple

def estimate_coeffs(design_matrix, values) -> list[float]:
    X = design_matrix
    beta = np.linalg.inv(X.T @ X) @ X.T @ values # formula from minimising the least-squares error
    return beta.flatten().tolist()[0]

def poly_approx(f, basis, sample_points: list, max_degree: int, dimension: int):
    indices = max_degree_indices(max_degree, dimension)
    assert len(sample_points) >= len(indices)
    design_matrix = np.matrix([
        [basis(idx)(x) for idx in indices] for x in sample_points
    ])
    values = [f(x) for x in sample_points]
    beta = estimate_coeffs(design_matrix, values)
    def approx(x): # approximation of f
        return sum(basis(idx)(x) * beta[i] for (i, idx) in enumerate(indices))
    return (approx, beta)

def max_degree_indices(max_degree: int, dimension: int): # list of all indices to be included as basis functions in the expansion, indices are unique and each one has sum <= max_degree 
    if dimension == 1:
        return [[i] for i in range(max_degree + 1)]
    
    if dimension == 2:
        indices = []
        for i in range(max_degree + 1):
            for j in range(max_degree + 1 - i):
                indices.append([i, j])
        return indices
    if dimension == 3:
        indices = []
        for i in range(max_degree + 1):
            for j in range(max_degree + 1 - i):
                for k in range(max_degree + 1 - i - j):
                    indices.append([i, j, k])
        return indices
    raise Exception("dimension must be 1, 2 or 3")

def chebshev_t(n: int, x: float):
    return math.cos(n * math.acos(x))

def cheb_basis(idx: list[int]): # tensor product chebyshev basis, each item of `idx` indicates the degree of the corresponding chebyshev polynomial
    return lambda x: math.prod(chebshev_t(idx[i], x[i]) for i in range(len(idx)))

def cheb_basis_scaled(rho: float): # scale chebyshev functions from domain [-1, 1]^n to [rho, 1]^n
    assert rho < 1
    def basis(idx: list[int]):
        return lambda x: cheb_basis(idx)([(1 - 2 * x[i] + rho)/(rho - 1) for i in range(len(idx))])
    return basis

def monomial_basis(n: int):
    return lambda x: x**n

def sin_or_cos(n: int, x: float):
    if n % 2 == 0:
        return math.cos(((n + 1) // 2) * x)
    return math.sin(((n + 1)//2) * x)

def trig_basis(idx: list[int]): # NB: to define up to maximum degree (frequency) m, this must be called with n = 2m
    return lambda x: math.prod(sin_or_cos(idx[i], x[i]) for i in range(len(idx)))

def annulus_basis(rho: float):
    assert rho < 1
    def basis(idx: list[int]): # tensor product basis of chebyshev polynomials for the radius coordinate and trigonometric polynomials for the angle coordinate. NB: as for trig_basis, to define up to a maximum degree (frequency) m, this must be called with n = 2m
        return lambda x: chebshev_t(idx[0], (1 - 2 * x[0] + rho)/(rho - 1)) * sin_or_cos(idx[1], x[1])
    return basis

def gen_cheb_points(n):
    return [math.cos(j * np.pi / n) for j in range(n + 1)]

def even_part(f):
    return lambda x: (f(x) + f(-x)) / 2 + np.cos(x) * 1e-5 * random.random()

def odd_part(f):
    return lambda x: (f(x) - f(-x)) / 2

NUM_GRID_POINTS = 10000

def analysis(f, domains: list[Tuple[float, float]], bases: list, max_degree: int, num_sample_points: int):
    dim = len(domains) # number of dimensions
    mesh = np.meshgrid(*[np.linspace(domain[0], domain[1], math.ceil(math.pow(NUM_GRID_POINTS, 1/dim))) for domain in domains])
    grid_points = np.vstack([points.ravel() for points in mesh]).T # equally spaced grid points with the correct dimension
    points_dict = {
        # "cheb": list(itertools.product(*[gen_cheb_points(math.ceil(math.pow(num_sample_points, 1/dim))) for _ in range(dim)])), # Cartesian product of chebyshev points
        # "random_cheb": random.sample(gen_cheb_points(2 * num_sample_points), num_sample_points),
        # "eq_space": np.linspace(-1, 1, num_sample_points).tolist(),
        "random_unif": [[random.random() * (domain[1] - domain[0]) + domain[0] for domain in domains] for _ in range(num_sample_points)]
    }
    for basis in bases:
        for (points_name, points) in points_dict.items():
            (approx, coeffs) = poly_approx(f, basis, points, max_degree, dimension=dim)
            max_error = max(np.abs(approx(x) - f(x)) for x in grid_points) # estimate of the max norm
            print(f"estimated max error (basis: {basis.__name__}, points: {points_name}):", max_error)

            even_coeffs = coeffs[::2]
            odd_coeffs = coeffs[1::2]
            # plt.scatter(range(0, max_degree + 1, 2), even_coeffs, marker="o")
            # plt.scatter(range(1, max_degree + 1, 2), odd_coeffs, marker="o")
            # plt.show()
            # plt.plot([x[0] for x in grid_points], [f(x) for x in grid_points], color="blue")
            # plt.plot([x[0] for x in grid_points], [approx(x) for x in grid_points], color="red")
            # plt.plot(np.linspace(-1, 1, 100), [f([x, 0]) for x in np.linspace(-1, 1, 100)], color="blue")
            # plt.plot(np.linspace(-1, 1, 100), [approx([x, 0]) for x in np.linspace(-1, 1, 100)], color="red")
            # plt.show()

max_degree = 10
num_sample_points = 500
rho = 1/4

def f(l: list[float]):
    x = l[0]
    return math.sin(x) + 1/4 * math.cos(20 * x)

def g(l: list[float]):
    [x, y] = l
    return x**2 + y**3 - 2*x*y - x**4 * y**3 + x**2 * y**5

def h(l: list[float]):
    [r, theta] = l
    return math.sin(theta)**3 / (1 + r**2)

# [-1, 1] cheb basis
analysis(f, domains=[(-1, 1)], bases=[cheb_basis], max_degree=max_degree, num_sample_points=num_sample_points)

# [rho, 1] scaled cheb basis
analysis(f, domains=[(rho, 1)], bases=[cheb_basis_scaled(rho)], max_degree=max_degree, num_sample_points=num_sample_points)

# [0, 2pi] trig basis
analysis(f, domains=[(0, np.pi)], bases=[trig_basis], max_degree=max_degree*2, num_sample_points=num_sample_points)

# [-1, 1]^2 (unit square) cheb tensor product basis
analysis(g, domains=[(-1, 1), (-1, 1)], bases=[cheb_basis], max_degree=max_degree, num_sample_points=num_sample_points)

# tensor product of chebyshev basis (for radius) and trig basis (for angle)
analysis(h, domains=[(rho, 1), (0, 2 * np.pi)], bases=[annulus_basis(rho)], max_degree=max_degree*2, num_sample_points=num_sample_points)

def non_smooth(l: list[float]):
    x = l[0]
    return max(math.sin(x), math.sin(2 * x), math.cos(x))

# non-smooth function approximated with chebyshev basis
analysis(non_smooth, domains=[(-1, 1)], bases=[cheb_basis], max_degree=100, num_sample_points=1000)

# progression: 1d [rho, 1] poly approx, 1d 2pi periodic trig polynomial approx, 2d square, annulus
# read up on condition number
# look up literature on deconvolution/polynomial division, why it's numerically unstable
# think about how to smooth out stacks of 1d surfaces (e.g. flippping the top surface, smooth out from remaining cusp)