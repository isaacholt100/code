import math
import numpy as np
from scipy.linalg import companion

def _esp_indices(max: int, length: int) -> list[list[int]]:
    assert length >= 1
    assert max >= length
    if length == 1:
        return list([i] for i in range(max))
    if max == length:
        return [list(range(max))]
    a = _esp_indices(max - 1, length)
    b = _esp_indices(max - 1, length - 1)
    for idx in b:
        idx.append(max - 1)
    return [*a, *b]

def elem_sym_poly(n: int, r: int):
    def fn(x: list[float]):
        assert len(x) == n
        if r > n:
            return 0.0
        if r == 0:
            return 1.0
        s = 0.0
        for idx in _esp_indices(n, r):
            s += math.prod(x[i] for i in idx)
        return s
    return fn

def reconstruct_roots(values):
    companion_column = [1] + [(-1)**(j + 1) * values[j] for j in range(len(values))]
    matrix = companion(companion_column) # scipy's Frobenius companion matrix implementation for simplicity
    roots = np.sort(np.linalg.eigvals(matrix))
    return roots