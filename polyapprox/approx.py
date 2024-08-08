from typing import Callable
import numpy as np

from bases import Basis

def estimate_coeffs(design_matrix, values: list[float]) -> list[float]:
    beta = np.linalg.lstsq(design_matrix, values, rcond=None)
    # print("beta:", beta[0])
    return beta[0]


def points_approx(basis: Basis, sample_points: list[list[float]], values: list[float], max_degree: int, dimension: int) -> tuple[Callable[..., float], list[float]]:
    assert len(sample_points) == len(values)
    indices = max_degree_indices(max_degree, dimension)
    assert len(sample_points) >= len(indices)
    design_matrix = np.matrix([
        [basis(idx)(x) for idx in indices] for x in sample_points
    ])
    print("X^T X:", np.abs(design_matrix.T @ design_matrix))
    beta = estimate_coeffs(design_matrix, values)

    def approx(*args):
        return sum(basis(idx)(list(args)) * beta[i] for (i, idx) in enumerate(indices))
    return (approx, beta)


def poly_approx(f: Callable[..., float], basis: Basis, sample_points: list[list[float]], max_degree: int, dimension: int):
    values = [f(*x) for x in sample_points]
    return points_approx(basis, sample_points, values, max_degree, dimension)


# list of all indices to be included as basis functions in the expansion, indices are unique and each one has sum <= max_degree
def max_degree_indices(max_degree: int, dimension: int) -> list[list[int]]:
    if dimension == 1:
        return [[i] for i in range(max_degree + 1)]

    indices = []
    for lower_index in max_degree_indices(max_degree, dimension - 1):
        s = sum(lower_index)
        for i in range(max_degree + 1 - s):
            indices.append([*lower_index.copy(), i])
    return indices
