import numpy as np

def estimate_coeffs(design_matrix, values: list[float]) -> list[float]:
    X = design_matrix
    beta = np.linalg.lstsq(X, values, rcond=None)
    return beta[0].tolist()

def points_approx(basis, sample_points: list, values: list[float], max_degree: int, dimension: int):
    indices = max_degree_indices(max_degree, dimension)
    assert len(sample_points) >= len(indices)
    design_matrix = np.matrix([
        [basis(idx)(x) for idx in indices] for x in sample_points
    ])
    beta = estimate_coeffs(design_matrix, values)
    def approx(x):
        return sum(basis(idx)(x) * beta[i] for (i, idx) in enumerate(indices))
    return (approx, beta, design_matrix)

def poly_approx(f, basis, sample_points: list, max_degree: int, dimension: int):
    values = [f(*x) for x in sample_points]
    return points_approx(basis, sample_points, values, max_degree, dimension)

def max_degree_indices(max_degree: int, dimension: int) -> list[list[int]]: # list of all indices to be included as basis functions in the expansion, indices are unique and each one has sum <= max_degree 
    if dimension == 1:
        return [[i] for i in range(max_degree + 1)]
    
    indices = []
    for lower_index in max_degree_indices(max_degree, dimension - 1):
        s = sum(lower_index)
        for i in range(max_degree + 1 - s):
            indices.append([*lower_index.copy(), i])
    return indices

