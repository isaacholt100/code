import math

import numpy as np
from scipy import stats
from custom_types import Curves, FloatArray

def curves_1(x: float):
    y1 = math.sin(x)
    y2 = math.cos(2*x)
    y3 = math.sin(2*x)
    y4 = 1.9+math.cos(x*np.pi/0.7)#1/3 + math.cos(2*x/3)
    y4 = 1/3 + math.cos(2*x/3)
    return sorted([y1, y2, y3, y4])[0:3]

def curves_2(x: float):
    return sorted([math.cos(x) + 1, math.sin(x/2) + 1, math.cos(3*x - 0.9) + 0.5])

def curves_3(x: float):
    return sorted([1/2*math.cos(x)+1/2, 1/2*math.sin(2*x) + 1/2, 1/2*math.sin(x)+1.1])

def curves_4(x: float):
    return sorted([math.cos(x) + 1, math.sin(x/2) + 1, math.cos(x - 0.9)])

def curves_sinh(a: float, b: float):
    def curves(x: float, y: float):
        return sorted([math.sinh(-math.sqrt(a**2 * y**2 + b**2 * x**2)/(a*b)), math.sinh(math.sqrt(a**2 * y**2 + b**2 * x**2)/(a*b)), 1e8])
    return curves

def _graphene_hamiltonian(kx,ky):
    a = 2.46 # Wikipedia
    t = 1
    k = np.array((kx, ky))
    d1 = (a/2) * np.array((1, np.sqrt(3)))
    d2 = (a/2) * np.array((1, -np.sqrt(3)))
    d3 = -a * np.array((1, 0))
    sx = np.matrix([[0, 1], [1, 0]]);
    sy = np.matrix([[0, -1j], [1j, 0]]);
    hx = np.cos(k@d1) + np.cos(k@d2) + np.cos(k@d3)
    hy = np.sin(k@d1) + np.sin(k@d2) + np.sin(k@d3)
    H = -t * (hx*sx - hy*sy)
    return H

def _g1(x, y):
    return math.cos(4*x) - math.sin(4*y) - 4
def _g2(x, y):
    return 2.5

def graphene_surfaces(x: float, y: float) -> list[float]:
    H = _graphene_hamiltonian(x, y)
    evals, _ = np.linalg.eigh(H)
    return sorted(evals.tolist() + [_g1(x, y), _g2(x, y)])[0:-1]

def sinusoid_surfaces(x: float, y: float) -> list[float]:
    a = 2*math.sin(6/5*(x+y))
    b = 2/3 - math.cos(x - y)
    c = 1.0
    d = 3/2 + math.sin((x + y)/3)
    return sorted([a, b, c, d])[0:-1]

_curves: list[Curves] = [curves_1, curves_2, curves_3, curves_4, curves_sinh(1, 1), graphene_surfaces, sinusoid_surfaces]

def surfaces_with_noise(stddev: float, curves: Curves) -> Curves:
    def noisy_curves(*args) -> list[float]:
        values = curves(*args)
        return [value + stats.norm.rvs(0, stddev) for value in values]
    return noisy_curves

def sample_values_with_noise(stddev: float, sample_values: FloatArray) -> FloatArray:
    assert sample_values.ndim == 2

    return sample_values + stats.norm.rvs(0, stddev, sample_values.shape)