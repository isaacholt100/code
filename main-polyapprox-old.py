import math
from matplotlib import pyplot as plt
import numpy as np
from polyapprox.bases import annulus_basis, cheb_basis, symmetric_annulus_basis, trig_basis, two_annulus_basis
from polyapprox.analyses import coefficient_convergence_analysis, depletion_probability_analysis

rho = 1/100000

## 1D Chebyshev ##

def f1(x):
    return math.sin(x) + 1/4 * math.cos(20 * x)

# [-1, 1] cheb basis
# analysis(f1, domains=[(-1, 1)], bases=[cheb_basis], max_degree=40, num_sample_points=100)

# [rho, 1] scaled cheb basis
# analysis(f1, domains=[(rho, 1)], bases=[cheb_basis_scaled(rho)], max_degree=20, num_sample_points=100)


## 1D Trig ##

def g2(theta):
    return math.cos(6*theta) + math.sin(6*theta) + math.cos(12*theta)

# [0, 2pi] trig basis
# analysis(g2, domains=[(0, 2*np.pi)], bases=[trig_basis], max_degree=20, num_sample_points=1000)


## Square ##

def f2(x, y):
    return x**2 + y**3 - 2*x*y - x**4 * y**3 + x**2 * y**5

# [-1, 1]^2 (unit square) cheb tensor product basis
# analysis(f2, domains=[(-1, 1), (-1, 1)], bases=[cheb_basis], max_degree=20, num_sample_points=500)


## Annulus ##

def f3(r, theta):
    return math.cos(3*theta) + math.cos(6*theta) * (r**3 - r**2) + r**6 - 2*r + math.cos(12*theta) + math.sin(6*theta) / (1 + r**2)

# coefficient_convergence_analysis(f3, 6, annulus_basis(rho), num_sample_points=1000, domains=[(rho, 1), (0, 2 * np.pi)], degrees=range(4, 30))

# tensor product of chebyshev basis (for radius) and trig basis (for angle)
# analysis(f3, domains=[(rho, 1), (0, 2 * np.pi)], bases=[annulus_basis(rho)], max_degree=23, num_sample_points=1000)

## 2D Trig ##

def g(theta1, theta2):
    return math.sin(theta1 - theta2) + math.cos(2*(theta1 - theta2)) + math.sin(3*(theta1 - theta2)) + math.sin(4*(theta1 - theta2))

# tensor product of trig bases
# analysis(g, domains=[(0, 2 * np.pi), (0, 2 * np.pi)], bases=[trig_basis], max_degree=14, num_sample_points=250)

## Tensor product of two annuli ##

def f4(r1, theta1, r2, theta2):
    return r1 + r2 * math.sin(theta1 - theta2) + r1 * math.cos((theta1 - theta2))

np.random.seed(42)

random_perturbation = np.random.random() * np.pi
# analysis(
#     f4,
#     domains=[(rho, 1), (0, 2 * np.pi), (rho, 1), (0, 2*np.pi)],
#     bases=[two_annulus_basis(rho)],
#     max_degree=10,
#     num_sample_points=1600,
#     perturbation = lambda x: [x[0], x[1] + random_perturbation, x[2], x[3] + random_perturbation]
# )

# read up on economisation

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

# depletion_probability_analysis(f3, domains=[(rho, 1), (0, 2 * np.pi)], basis=annulus_basis(rho), max_degree=23, num_sample_points=1000, inclusion_probabilities=np.linspace(0, 1, 30))

# TODO: look up paper(s) by Sergey N. Pozdnyakov, Michele Ceriotti
# TODO: try post-symmetrisation
# TODO: try using loads of rotations, not much data

# TODO: plot 1d cut of the approximation of g with ~40 rotations vs actual g (cut along theta1 = -theta2 plane)