import math
import numpy as np
from polyapprox.bases import annulus_basis, symmetric_annulus_basis, two_annulus_basis
from polyapprox.analyses import max_mean_error_analysis, analysis

rho = 1/100000

## 1D Chebyshev ##

def f1(x):
    return math.sin(x) + 1/4 * math.cos(20 * x)

# [-1, 1] cheb basis
# analysis(f1, domains=[(-1, 1)], bases=[cheb_basis], max_degree=40, num_sample_points=100)

# [rho, 1] scaled cheb basis
# analysis(f1, domains=[(rho, 1)], bases=[cheb_basis_scaled(rho)], max_degree=20, num_sample_points=100)


## 1D Trig ##

# [0, 2pi] trig basis
# analysis(f1, domains=[(0, np.pi)], bases=[trig_basis], max_degree=40*2, num_sample_points=100)


## Square ##

def f2(x, y):
    return x**2 + y**3 - 2*x*y - x**4 * y**3 + x**2 * y**5

# [-1, 1]^2 (unit square) cheb tensor product basis
# analysis(f2, domains=[(-1, 1), (-1, 1)], bases=[cheb_basis], max_degree=20, num_sample_points=500)


## Annulus ##

def f3(r, theta):
    return math.cos(3*theta) * (r**3 - r**2) + r**6 - 2*r + math.cos(12*theta) + math.sin(6*theta) / (1 + r**2)

# tensor product of chebyshev basis (for radius) and trig basis (for angle)
# analysis(f3, domains=[(rho, 1), (0, 2 * np.pi)], bases=[annulus_basis(rho), symmetric_annulus_basis(rho)], max_degree=20, num_sample_points=1000)


## Tensor product of two annuli ##

def f4(r1, theta1, r2, theta2):
    return r1 + r2 * math.sin(theta1 - theta2) + r1 * math.cos(6 * (theta1 - theta2))

random_perturbation = np.random.random() * np.pi
analysis(
    f4,
    domains=[(rho, 1), (0, 2 * np.pi), (rho, 1), (0, np.pi)],
    bases=[two_annulus_basis(rho)],
    max_degree=10,
    num_sample_points=1600,
    perturbation = lambda x: [x[0], x[1] + random_perturbation, x[2], x[3] + random_perturbation]
)


## 1D Non-smooth ##

def f5(x):
    return max(math.sin(x), math.sin(2 * x), math.cos(x))

def f6(x):
    return 1/4 * (abs(x - 1/2) - abs(x) + abs(x - 0.8))

# analysis(f5, domains=[(-1, 1)], bases=[cheb_basis], max_degree=20, num_sample_points=500)
# analysis(f6, domains=[(-1, 1)], bases=[cheb_basis], max_degree=100, num_sample_points=1000)

def test_fn_1(epsilon: float):
    return lambda r, theta: math.cos(3*theta) * (r**3 - r**2) + r**6 - 2*r + math.cos(12*theta) + math.sin(6*theta) / (1 + epsilon*math.sin(3*theta)**2 + r**2)

# max_mean_error_analysis(test_fn_1, 25, 1000, basis=symmetric_annulus_basis(rho))

# read up on condition number
# look up literature on deconvolution/polynomial division, why it's numerically unstable
# think about how to smooth out stacks of 1d surfaces (e.g. flippping the top surface, smooth out from remaining cusp)
# reading: ch 13 book, ch 3 and 4 book, ch 7 and 8 book

# TODO: plot max and mean errors of coeffs (coeffs from original sample versus coeffs from rotated) as epsilon varies from 1e-10 to 1
# TODO: define another annulus basis that only includes 2pi/3 rotation invariant sin/cos basis functions, so e.g. 1 -> sin(3x), 2 -> cos(3x), 3 -> sin(6x). with same degree of freedom, and same random samples, which is more accurate? can then compare non-symmetric basis and one with rotated points with invariant basis
# TODO: next do tensor product of two annuli