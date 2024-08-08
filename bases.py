import math
from typing import Callable
import numpy as np

BasisFunction = Callable[[list[float]], float]
Basis = Callable[[list[int]], BasisFunction]

def chebyshev_t(n: int, x: float) -> float:
    return math.cos(n * math.acos(x))

def cheb_basis(idx: list[int]) -> BasisFunction: # tensor product chebyshev basis, each item of `idx` indicates the degree of the corresponding chebyshev polynomial
    return lambda x: math.prod(chebyshev_t(idx[i], x[i]) for i in range(len(idx)))

def cheb_basis_scaled(rho: float) -> Basis: # scale chebyshev functions from domain [-1, 1]^n to [rho, 1]^n
    assert rho < 1
    def basis(idx: list[int]) -> BasisFunction:
        return lambda x: cheb_basis(idx)([np.clip((1 - 2 * x[i] + rho)/(rho - 1), -1, 1) for i in range(len(idx))]) # rescale and clamp between -1 and 1 to prevent domain error
    return basis

def sin_or_cos(n: int, x: float) -> float:
    if n % 2 == 0:
        return math.cos(((n + 1) // 2) * x)
    return math.sin(((n + 1)//2) * x)

def trig_basis(idx: list[int]) -> BasisFunction: # NB: to define up to maximum degree (frequency) m, this must be called with n = 2m
    return lambda x: math.prod(sin_or_cos(idx[i], x[i]) for i in range(len(idx)))

def exp_or_minus(n: int, x: float) -> float:
    if n % 2 == 0:
        return np.exp(1j*((n + 1)//2)*x)
    return np.exp(-1j*((n + 1)//2)*x)

def exponential_basis(idx: list[int]) -> BasisFunction:
    return lambda x: math.prod(exp_or_minus(idx[i], x[i]) for i in range(len(idx)))

# trig basis normalised to have l2[0, 2pi] norm 1
def trig_basis_l2_normalised(idx: list[int]) -> BasisFunction: # NB: to define up to maximum degree (frequency) m, this must be called with n = 2m
    def sin_or_cos_normalised(n: int, x: float):
        if n == 0:
            return 1/(2*np.pi)
        return sin_or_cos(n, x)/np.pi
    return lambda x: math.prod(sin_or_cos_normalised(idx[i], x[i]) for i in range(len(idx)))

def annulus_basis(rho: float) -> Basis:
    assert rho < 1
    def basis(idx: list[int]) -> BasisFunction: # tensor product basis of chebyshev polynomials for the radius coordinate and trigonometric polynomials for the angle coordinate. NB: as for trig_basis, to define up to a maximum degree (frequency) m, this must be called with n = 2m
        return lambda x: chebyshev_t(idx[0], np.clip((1 - 2 * x[0] + rho)/(rho - 1), -1, 1)) * sin_or_cos(idx[1], x[1]) # rescale and clamp between -1 and 1 to prevent domain error
    return basis

def symmetric_annulus_basis(rho: float) -> Basis:
    assert rho < 1
    def basis(idx: list[int]) -> BasisFunction:
        return lambda x: chebyshev_t(idx[0], np.clip((1 - 2 * x[0] + rho)/(rho - 1), -1, 1)) * sin_or_cos(idx[1], 3*x[1])
    return basis

def two_annulus_basis(rho: float) -> Basis:
    assert rho < 1
    def basis(idx: list[int]) -> BasisFunction:
        return lambda x: annulus_basis(rho)(idx[0:2])([x[0], x[1]]) * annulus_basis(rho)(idx[2:4])([x[2], x[3]])
    return basis