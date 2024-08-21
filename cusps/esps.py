import math
from typing import Callable
import numpy as np
from scipy.linalg import companion

from custom_types import ComplexArray

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

def elem_sym_poly(n: int, r: int) -> Callable[[list[float]], float]:
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

def reconstruct_roots_frob(values: list[float]) -> ComplexArray:
    companion_column = [(-1)**(j + 1) * values[j] for j in range(len(values))]
    companion_column.insert(0, 1.0)
    matrix = companion(companion_column)
    roots = np.sort(np.linalg.eigvals(matrix))
    return roots

def reconstruct_roots(values: list[float]) -> ComplexArray:
    return reconstruct_roots_chebyshev(values)

def reconstruct_roots_chebyshev(values: list[float]) -> ComplexArray:
    n = len(values)
    if n == 3:
        c = [-(2*values[0] + 4*values[2]), 4*values[1] + 3, -2*values[0], 1]
        return np.polynomial.chebyshev.chebroots(c)
    if n == 2:
        c = [1 + 2*values[1], -2*values[0], 1]
        return np.polynomial.chebyshev.chebroots(c)
    raise Exception(f"invalid value of n: {n}")