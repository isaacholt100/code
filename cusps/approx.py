from typing import Any, Callable
import numpy as np

from bases import Basis
from custom_types import FloatArray

def max_degree_indices(max_degree: int, dimension: int) -> list[list[int]]: # list of all indices to be included as basis functions in the expansion, indices are unique and each one has sum <= max_degree 
    if dimension == 1:
        return [[i] for i in range(max_degree + 1)]
    
    indices = []
    for lower_index in max_degree_indices(max_degree, dimension - 1):
        s = sum(lower_index)
        for i in range(max_degree + 1 - s):
            indices.append([*lower_index.copy(), i])
    return indices

def estimate_coeffs(design_matrix: FloatArray, values: FloatArray) -> FloatArray:
    assert design_matrix.ndim == 2
    assert values.ndim == 1
    assert design_matrix.shape[0] == values.shape[0]
    
    X = design_matrix
    beta = np.linalg.lstsq(X, values, rcond=None)
    return beta[0]

def points_approx(
    basis: Basis,
    sample_points: FloatArray,
    values: FloatArray,
    max_degree: int
) -> tuple[Callable[[list[float]], float], FloatArray]:
    assert sample_points.ndim == 2
    assert values.ndim == 1
    assert sample_points.shape[0] == values.shape[0]

    dimension = sample_points.shape[1]
    indices = max_degree_indices(max_degree, dimension)
    # assert len(sample_points) >= len(indices)
    design_matrix = np.matrix([
        [basis(idx)(x) for idx in indices] for x in sample_points
    ])
    beta = estimate_coeffs(design_matrix, values)
    def approx(x: list[float]):
        return sum(basis(idx)(x) * beta[i] for (i, idx) in enumerate(indices))
    return (approx, beta)