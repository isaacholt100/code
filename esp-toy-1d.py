import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import companion
import math
import random

np.random.seed(42) # make things reproducible

# Convenience function to construct intersecting sinusoidal stack described in the paper
def intersecting_cosine_curves_1(x):
    y1 = np.sin(x)
    y2 = np.cos(2*x)
    y3 = np.sin(2*x)
    y4 = 1/3+np.cos(2*x/3)
    return [y1, y2, y3, y4]

def estimate_coeffs(design_matrix, values) -> list[float]:
    X = design_matrix
    beta = np.linalg.inv(X.T @ X) @ X.T @ values # formula from minimising the least-squares error
    return beta.flatten().tolist()[0]

def points_approx(basis, sample_points: list, values: list[float], max_degree: int):
    indices = range(max_degree + 1)
    assert len(sample_points) >= len(indices)
    design_matrix = np.matrix([
        [basis(idx)(x) for idx in indices] for x in sample_points
    ])
    beta = estimate_coeffs(design_matrix, values)
    def approx(x):
        return sum(basis(idx)(x) * beta[i] for (i, idx) in enumerate(indices))
    return (approx, beta, design_matrix)

def chebyshev_t(n: int, x: float):
    return math.cos(n * math.acos(x))

def cheb_basis_shifted(a: float, b: float):
    return lambda n: lambda x: chebyshev_t(n, 2*(x - a)/(b - a) - 1)

def esp_indices(max: int, length: int) -> list[list[int]]:
    assert length >= 1
    assert max >= length
    if length == 1:
        return list([i] for i in range(max))
    if max == length:
        return [list(range(max))]
    a = esp_indices(max - 1, length)
    b = esp_indices(max - 1, length - 1)
    for idx in b:
        idx.append(max - 1)
    return [*a, *b]

def esp(n: int, r: int):
    def fn(x: list[float]):
        if r > n:
            return 0
        if r == 0:
            return 1
        s = 0
        for idx in esp_indices(n, r):
            s += math.prod(x[i] for i in idx)
        return s
    return fn

def chebyshev_sample(n: int, points: list[float], a, b):
    x_sample = []
    for x in points:
        k = ((2*n + 1)/math.pi * math.acos(2/(b - a)*(x - (b + a)/2)) - 1) / 2
        assert 0 <= k and k <= n
        # x_sample.append(x)
        # y_sample.append(y)
        if x < 2*abs(k - n/2)/n:
            x_sample.append(x)
    return x_sample

def esp_derivs():
    delta_x = 1e-6
    delta = 0.1
    b = 2
    matched_cusp_points = [0.38, 1.05]
    c = matched_cusp_points[1] + delta/2
    d = c# + delta/2
    x_values = np.linspace(0, 2, round((2 - 0)/delta_x))
    y_values = np.partition(np.vstack(intersecting_cosine_curves_1(x_values)), [0, 1, 2, 3], axis=0)

    x_sample = []
    y_sample = []
    modified_indices = []
    for i, (x, y) in enumerate(zip(x_values, y_values[2, :])):
        if c <= x and x <= b and (x <= 1.25 or x >= 1.75):
            # if x - c < 1e-2:
            #     for _ in range(5):
            #         x_sample.append(x)
            #         y_sample.append(y)
            # n = 50
            # k = ((2*n + 1)/math.pi * math.acos(2/(b - c)*(x - (b + c)/2)) - 1) / 2
            # # x_sample.append(x)
            # # y_sample.append(y)
            # if random.random() < 2*abs(k - n/2)/n:
            x_sample.append(x)
            y_sample.append(y)
            modified_indices.append(i)
    (approx, _, _) = points_approx(cheb_basis_shifted(c, b), [*x_sample], [*y_sample], max_degree=8)
    x_modified = x_values[modified_indices[0]:modified_indices[-1] + 1]
    y_modified = [approx(x) for x in x_modified]
    for i in range(modified_indices[0], modified_indices[-1] + 1):
        if d <= x:
            # break
            y_values[2, i] = y_modified[i - modified_indices[0]]
    
    # unsorted = 
    sum_curve = y_values[0, :] + y_values[1, :] + y_values[2, :]
    mixed_curve = y_values[0, :] * y_values[1, :] + y_values[0, :] * y_values[2, :] + y_values[1, :] * y_values[2, :]
    product_curve = y_values[0, :] * y_values[1, :] * y_values[2, :]
    plt.plot(x_values, sum_curve)
    plt.plot(x_values, mixed_curve)
    plt.plot(x_values, product_curve)
    plt.show()
    derivs1 = (sum_curve[1:] - sum_curve[0:-1]) / delta_x
    derivs2 = (mixed_curve[1:] - mixed_curve[0:-1]) / delta_x
    derivs3 = (product_curve[1:] - product_curve[0:-1]) / delta_x
    plt.plot(x_values[0:-1], derivs1)
    plt.plot(x_values[0:-1], derivs2)
    plt.plot(x_values[0:-1], derivs3)
    plt.show()

esp_derivs()
# exit(0)
# a = -1
# b = 1
# n = 100
# plt.scatter([math.cos((2*k + 1)*math.pi / (2*(n + 1))) for k in range(n + 1)], [0]*(n + 1))
# rand_unif_points = [random.random() * (b - a) + a for _ in range(n)]
# # plt.scatter(rand_unif_points, [0]*len(rand_unif_points))
# random_cheb_points = chebyshev_sample(10000, rand_unif_points, a, b)
# plt.scatter(random_cheb_points, [0]*len(random_cheb_points))
# plt.show()
# exit(0)

# Generate x values
def reconstruct(a: float, b: float, sample_size: int, curves, interpolant_degree: int, smoothing_degree: int):
    x_values = np.sort(np.random.uniform(a, b, sample_size))  # uniformly distributed random samples
    y_values = curves(x_values)
    sorted_curves = np.partition(np.vstack(y_values), list(range(len(y_values))), axis=0)

    if smoothing_degree > 0:
        delta = 0.2
        matched_cusp_points = [0.38, 1.05]
        c = matched_cusp_points[1] + delta/2
        d = c + delta

        x_sample = []
        y_sample = []
        modified_indices = []
        for i, (x, y) in enumerate(zip(x_values, sorted_curves[len(y_values) - 2, :])):
            if c <= x and x <= b and (x <= 1.25 or x >= 1.75):
                # if x - c < 1e-2:
                #     for _ in range(5):
                #         x_sample.append(x)
                #         y_sample.append(y)
                # n = 50
                # k = ((2*n + 1)/math.pi * math.acos(2/(b - c)*(x - (b + c)/2)) - 1) / 2
                # # x_sample.append(x)
                # # y_sample.append(y)
                # if random.random() < 2*abs(k - n/2)/n:
                x_sample.append(x)
                y_sample.append(y)
                modified_indices.append(i)
        (approx, _, _) = points_approx(cheb_basis_shifted(c, b), [*x_sample], [*y_sample], max_degree=smoothing_degree)
        x_modified = x_values[modified_indices[0]:modified_indices[-1] + 1]
        y_modified = [approx(x) for x in x_modified]
        for i in range(modified_indices[0], modified_indices[-1] + 1):
            epsilon = 5e-1
            if c - epsilon <= x and x <= c + epsilon:
                np.delete(x_values, i)
                np.delete(sorted_curves, i, axis=1)
            elif d <= x:
                # break
                sorted_curves[len(y_values) - 2, i] = y_modified[i - modified_indices[0]]
        # plt.plot(x_values, sorted_curves[len(y_values) - 2, :])

    esps = [esp(len(y_values) - 1, r) for r in range(1, len(y_values))]

    

    # Compute the ESPs
    esp_curves = [[esps[j](sorted_curves[:, i].tolist()) for i in range(len(x_values))] for j in range(len(y_values) - 1)]

    from numpy.polynomial.chebyshev import chebfit as polyfit
    from numpy.polynomial.chebyshev import chebval as polyval

    interpolated_curves = [polyval(x_values, polyfit(x_values, esp_curve, interpolant_degree)) for esp_curve in esp_curves] # The interpolated things are stand-ins for what a ML model would in practice give us

    points = [] # We could pre-allocate since the dimensions are known but for small problems efficiency isn't really a concern
    for i, values in enumerate(zip(*interpolated_curves)):
        companion_column = [1]
        for i in range(len(values)):
            companion_column.append((-1)**(i + 1) * values[i])
        matrix = companion(companion_column) # scipy's Frobenius companion matrix implementation for simplicity
        roots = np.sort(np.linalg.eigvals(matrix))
        points.append(roots)

    points = np.array(points).T

    max_abs_errors = [max(abs(points[j, i] - sorted_curves[j, i]) for i in range(len(x_values))) for j in range(len(y_values) - 2)]

    return x_values, sorted_curves, points, max_abs_errors

x_values, sorted_curves, points, max_abs_errors = reconstruct(a=0, b=2, sample_size=4000, curves=intersecting_cosine_curves_1, interpolant_degree=20, smoothing_degree=8)
# Make the default color wheel accessible
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure(0)
plt.plot(x_values, sorted_curves[0, :], color=colors[0])
plt.plot(x_values, sorted_curves[1, :], color=colors[1])
plt.plot(x_values, sorted_curves[2, :], color=colors[2])
# plt.plot(x_values, sorted_curves[3, :], color=colors[3], linewidth=4) # optionally plot the missing fourth stack
plt.title("Original multi-surface data")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

plt.figure(1)
plt.plot(x_values, points[0, :], color=colors[0])
plt.plot(x_values, points[1, :], color=colors[1])
plt.plot(x_values, points[2, :], color=colors[2])
plt.title("ESP-reconstructed multi-surface")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

errors0 = np.abs(points[0, :] - sorted_curves[0, :])
errors1 = np.abs(points[1, :] - sorted_curves[1, :])
plt.plot(x_values, np.log10(errors0))
plt.plot(x_values, np.log10(errors1))
plt.show()

print("sum max abs error:", sum(max_abs_errors))
interpolant_degree = 20
_, _, _, errors_without_smoothing = reconstruct(a=0, b=2, sample_size=4000, curves=intersecting_cosine_curves_1, interpolant_degree=interpolant_degree, smoothing_degree=0)
min_error, min_error_degree = sum(errors_without_smoothing), 0
for j in range(4, 15):
    _, _, _, max_abs_errors = reconstruct(a=0, b=2, sample_size=4000, curves=intersecting_cosine_curves_1, interpolant_degree=interpolant_degree, smoothing_degree=j)
    e = sum(max_abs_errors)
    if e < min_error:
        min_error_degrees = j
        min_error = e

print("error without smoothing:", math.log(sum(errors_without_smoothing)), sum(errors_without_smoothing))
print("min error and associated degree:", math.log(min_error), min_error, min_error_degrees)

exit(0)
points = []
errors = []
min_error, min_error_degrees = sum(errors_without_smoothing), (0, 0)
for i in range(10, 50, 2):
    for j in [0, *range(4, 25)]:
        points.append((i, j))
        _, _, _, max_abs_errors = reconstruct(a=0, b=2, sample_size=4000, curves=intersecting_cosine_curves_1, interpolant_degree=i, smoothing_degree=j)
        e = sum(max_abs_errors)
        if e < min_error:
            min_error_degrees = (i, j)
            min_error = e
        errors.append(math.log(e))

scatter_plot = plt.scatter(*list(zip(*points)), c=errors, cmap="viridis")
plt.xlabel("esp interpolant degree")
plt.ylabel("smoothing degree")
plt.colorbar(scatter_plot, label="log sum max abs error")
plt.show()
print("min error and associated degrees:", min_error, min_error_degrees)

# TODO: IDEA: use Hermite interpolation/just look at derivative around matched cusp point, construct polynomial which matches this derivative (and ofc matches the values at the matched cusp points