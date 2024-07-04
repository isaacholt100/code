import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import companion
import math
import random
from numpy.polynomial.chebyshev import chebfit as polyfit
from numpy.polynomial.chebyshev import chebval as polyval

np.random.seed(42) # make things reproducible

def intersecting_cosine_curves_1(x):
    y1 = math.sin(x)
    y2 = math.cos(2*x)
    y3 = math.sin(2*x)
    y4 = 1/3 + math.cos(2*x/3)
    return np.array(sorted([y1, y2, y3, y4])[0:3])

def gap_weighted_error(surface, approximation):
    def delta(s, i: int, j: int):
        return s[i] - s[j]
    epsilon_w = 5e-2
    max_error = 0
    for i in range(len(approximation)):
        for j in range(len(approximation)):
            if i == j:
                continue
            m = max(abs(delta(approximation, i, j) - delta(surface, i, j))/(epsilon_w + abs(delta(surface, i, j))))
            if m > max_error:
                max_error = m
    return max_error

def estimate_coeffs(design_matrix, values) -> list[float]:
    X = design_matrix
    beta = np.linalg.lstsq(X, values, rcond=None)
    return beta[0].tolist()

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
        assert len(x) == n
        if r > n:
            return 0.0
        if r == 0:
            return 1.0
        s = 0.0
        for idx in esp_indices(n, r):
            s += math.prod(x[i] for i in idx)
        return s
    return fn

# print(esp(4, 2)([1, 2, 3, 4]))
# exit(0)

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
    delta_x = 1e-4
    delta = 0# 0.1
    b = 2
    matched_cusp_points = [0.38, np.pi/3]
    c = matched_cusp_points[1]
    d = c
    x_values = [0.0]
    while x_values[-1] <= 2 - delta_x:
        x_values.append(x_values[-1] + delta_x)
    x_values = np.array(x_values)
    y_values = intersecting_cosine_curves_1(x_values)

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
    ## TODO: try adding derivative condition
    x_modified = x_values[modified_indices[0]:modified_indices[-1] + 1]
    y_modified = [approx(x) for x in x_modified]
    for i in range(modified_indices[0], modified_indices[-1] + 1):
        if d <= x:
            # break
            y_values[2, i] = y_modified[i - modified_indices[0]]
    
    # i = 0
    # while True:
    #     if i >= len(x_values):
    #         break
    #     x = x_values[i]
    #     epsilon = 0.05
    #     if c - epsilon <= x and x <= c + epsilon:
    #         # i += 1; continue
    #         x_values = np.delete(x_values, i)
    #         y_values = np.delete(y_values, i, axis=1)
    #     i += 1
    # x_values_2 = np.linspace(0.95, 2, round(0.2 / delta_x))
    # y_values_2 = np.array([sorted(intersecting_cosine_curves_1(x))[2] for x in x_values_2])
    plt.plot(x_values, y_values[2, :])
    plt.show()
    derivs = (y_values[2, 1:] - y_values[2, 0:-1]) / delta_x
    plt.plot(x_values[0:-1], derivs)
    plt.show()
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
    for i, d in enumerate(derivs1):
        if abs(d) > 10:
            print("large deriv between:", sum_curve[i+1] - sum_curve[i], "at index", i)
    plt.plot(x_values[0:-1], derivs1)
    plt.plot(x_values[0:-1], derivs2)
    plt.plot(x_values[0:-1], derivs3)
    plt.show()

# esp_derivs()
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
def reconstruct(a: float, b: float, sample_size: int, curves, interpolant_degree: int, smoothing_degree: int, delta_p: float, matched_cusp_points: list[float], removal_epsilon: float):
    x_values = np.sort(np.random.uniform(a, b, sample_size)) # uniformly distributed random samples
    sorted_curves = np.array([curves(x) for x in x_values]).T

    if smoothing_degree > 0: # if smoothing degree is 0, do not attempt to smooth out the unmatched cusp
        delta = 0.2
        c = np.pi/3
        d = c + 0.1

        x_sample = []
        y_sample = []
        modified_indices = []
        for i, (x, y) in enumerate(zip(x_values, sorted_curves[-1, :])):
            if c <= x and x <= b and (x <= 1.25 or x >= 1.75):
                x_sample.append(x)
                y_sample.append(y)
                modified_indices.append(i)
        (approx, _, _) = points_approx(cheb_basis_shifted(c, b), [*x_sample], [*y_sample], max_degree=smoothing_degree)
        x_modified = x_values[modified_indices[0]:modified_indices[-1] + 1]
        y_modified = [approx(x) for x in x_modified]
        for i in range(modified_indices[0], modified_indices[-1] + 1):
            if d <= x:
                # break
                sorted_curves[-1, i] = y_modified[i - modified_indices[0]]

        if removal_epsilon > 0: # if removal epsilon is 0, do not remove points around where we start approximating
            # remove points near where we start approximating the curve
            i = 0
            while True:
                if i >= len(x_values):
                    break
                x = x_values[i]
                epsilon = 0.05
                if c - epsilon <= x and x <= c + epsilon:
                    x_values = np.delete(x_values, i)
                    sorted_curves = np.delete(sorted_curves, i, axis=1)
                i += 1

    # esps = [esp(len(sorted_curves), r) for r in range(1, len(sorted_curves) + 1)]
    # esp_curves = [[esp(sorted_curves[:, i].tolist()) for i in range(len(x_values))] for esp in esps]

    # interpolated_curves = [polyval(x_values, polyfit(x_values, esp_curve, interpolant_degree)) for esp_curve in esp_curves] # The interpolated things are stand-ins for what a ML model would in practice give us
    # plt.plot(x_values, interpolated_curves[0])
    # plt.plot(x_values, interpolated_curves[1])
    # plt.plot(x_values, interpolated_curves[2])
    # plt.title("ESPs")
    # plt.show()

    # points = []
    # for i, values in enumerate(zip(*interpolated_curves)):
    #     companion_column = [1] + [(-1)**(j + 1) * values[j] for j in range(len(values))]
    #     matrix = companion(companion_column) # scipy's Frobenius companion matrix implementation for simplicity
    #     roots = np.sort(np.linalg.eigvals(matrix))
    #     points.append(roots)

    # points = np.array(points).T

    away_from_cusp_indices = []
    near_cusp_indices = []
    for i in range(len(x_values)):
        near = False
        for p in matched_cusp_points:
            if abs(x_values[i] - p) < delta_p:
                near_cusp_indices.append(i)
                near = True
                break
        if not near:
            away_from_cusp_indices.append(i)
    assert set(away_from_cusp_indices).union(set(near_cusp_indices)) == set(range(len(x_values)))
    esps_away = [esp(len(sorted_curves) - 1, r) for r in range(1, len(sorted_curves))]
    esps_near = [esp(len(sorted_curves), r) for r in range(1, len(sorted_curves) + 1)]
    x_values_away = [x_values[i] for i in away_from_cusp_indices]
    x_values_near = [x_values[i] for i in near_cusp_indices]
    # Compute the ESPs
    esp_away_curves = [[esps_away[j](sorted_curves[0:-1, i].tolist()) for i in away_from_cusp_indices] for j in range(len(sorted_curves) - 1)]
    esp_near_curves = [[esps_near[j](sorted_curves[:, i].tolist()) for i in near_cusp_indices] for j in range(len(sorted_curves))]

    interpolated_curves_away = [polyval(x_values_away, polyfit(x_values_away, esp_curve, interpolant_degree)) for esp_curve in esp_away_curves] if len(x_values_away) > 0 else [] # The interpolated things are stand-ins for what a ML model would in practice give us
    interpolated_curves_near = [polyval(x_values_near, polyfit(x_values_near, esp_curve, interpolant_degree)) for esp_curve in esp_near_curves] if len(x_values_near) > 0 else [] # The interpolated things are stand-ins for what a ML model would in practice give us

    points = [None] * len(x_values) # We could pre-allocate since the dimensions are known but for small problems efficiency isn't really a concern
    for i, values in enumerate(zip(*interpolated_curves_away)):
        companion_column = [1] + [(-1)**(j + 1) * values[j] for j in range(len(values))]
        matrix = companion(companion_column) # scipy's Frobenius companion matrix implementation for simplicity
        roots = np.sort(np.linalg.eigvals(matrix))
        idx = away_from_cusp_indices[i]
        points[idx] = [*roots, math.nan]
    for i, values in enumerate(zip(*interpolated_curves_near)):
        companion_column = [1] + [(-1)**(j + 1) * values[j] for j in range(len(values))]
        matrix = companion(companion_column) # scipy's Frobenius companion matrix implementation for simplicity
        roots = np.sort(np.linalg.eigvals(matrix))
        idx = near_cusp_indices[i]
        points[idx] = roots

    points = np.array(points).T

    max_abs_errors = [max(abs(points[j, i] - sorted_curves[j, i]) for i in range(len(x_values))) for j in range(len(sorted_curves) - 1)]

    return x_values, sorted_curves, points, max_abs_errors

def plot_delta_p():
    delta_p_values = np.linspace(0.1, 0.3, 100)
    max_abs_errors, gap_weighted_errors = [], []
    for delta_p in delta_p_values:
        _, sorted_curves, points, e = reconstruct(a=0, b=2, sample_size=4000, curves=intersecting_cosine_curves_1, interpolant_degree=20, smoothing_degree=0, delta_p=delta_p, matched_cusp_points=[np.pi/8, np.pi/6, np.pi/3, np.pi*5/8], removal_epsilon=0.05)
        max_abs_errors.append(sum(e))
        gap_weighted_errors.append(gap_weighted_error(sorted_curves, points))

    plt.plot(delta_p_values, max_abs_errors)
    plt.ylabel("max abs error")
    plt.xlabel("delta_p")
    plt.show()
    plt.plot(delta_p_values, gap_weighted_errors)
    plt.ylabel("gap weighted error")
    plt.xlabel("delta_p")
    plt.show()
# plot_delta_p()

x_values, sorted_curves, points, max_abs_errors = reconstruct(a=0, b=2, sample_size=4000, curves=intersecting_cosine_curves_1, interpolant_degree=20, smoothing_degree=8, delta_p=0.2883, matched_cusp_points=[np.pi/8, np.pi/6, np.pi/3, np.pi*5/8], removal_epsilon=0.05)
# 0.35633, 0.35755, 0.35899

plt.figure(0)
plt.plot(x_values, sorted_curves[0, :])
plt.plot(x_values, sorted_curves[1, :])
plt.plot(x_values, sorted_curves[2, :])
# plt.plot(x_values, sorted_curves[3, :], color=colors[3], linewidth=4) # optionally plot the missing fourth stack
plt.title("Original multi-surface data")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

plt.figure(1)
plt.plot(x_values, points[0, :])
plt.plot(x_values, points[1, :])
plt.plot(x_values, points[2, :])
plt.title("ESP-reconstructed multi-surface")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

# errors0 = np.abs(points[0, :] - sorted_curves[0, :])
# errors1 = np.abs(points[1, :] - sorted_curves[1, :])
# plt.plot(x_values, np.log10(errors0))
# plt.plot(x_values, np.log10(errors1))
# plt.show()

print("sum max abs error:", sum(max_abs_errors))
print("gap weighted error:", gap_weighted_error(sorted_curves, points))
exit(0)
# interpolant_degree = 20
# _, _, _, errors_without_smoothing = reconstruct(a=0, b=2, sample_size=4000, curves=intersecting_cosine_curves_1, interpolant_degree=interpolant_degree, smoothing_degree=0)
# min_error, min_error_degree = sum(errors_without_smoothing), 0
# for j in range(4, 15):
#     _, _, _, max_abs_errors = reconstruct(a=0, b=2, sample_size=4000, curves=intersecting_cosine_curves_1, interpolant_degree=interpolant_degree, smoothing_degree=j)
#     e = sum(max_abs_errors)
#     if e < min_error:
#         min_error_degrees = j
#         min_error = e

# print("error without smoothing:", math.log(sum(errors_without_smoothing)), sum(errors_without_smoothing))
# print("min error and associated degree:", math.log(min_error), min_error, min_error_degrees)
