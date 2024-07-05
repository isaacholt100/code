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

def max_abs_errors(original, reconstructed):
    return [max(abs(reconstructed[j, i] - original[j, i]) for i in range(reconstructed.shape[1])) for j in range(len(original) - 1)]

def sum_max_abs_error(original, reconstructed):
    return sum(max_abs_errors(original, reconstructed))

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

def elem_sym_poly(n: int, r: int):
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

def rand_unif_samples(a: float, b: float, sample_size: int):
    assert sample_size > 0
    return np.sort(np.random.uniform(a, b, sample_size))

def reconstruct_enhanced(x_values, curves, interpolant_degree: int, smoothing_degree: int, delta_p: float, matched_cusp_points: list[float], removal_epsilon: float, smoothing_sample_range: float):
    assert interpolant_degree > 0
    assert smoothing_degree >= 0
    assert delta_p >= 0
    assert removal_epsilon >= 0

    original_surfaces = np.array([curves(x) for x in x_values]).T
    surface_count = original_surfaces.shape[0]

    if smoothing_degree > 0: # if smoothing degree is 0, do not attempt to smooth out the unmatched cusp
        for (start, end) in zip(matched_cusp_points[0:-1], matched_cusp_points[1:]):
            start = start
            end = end
            x_sample = []
            y_sample = []
            modified_indices = []
            for i, (x, y) in enumerate(zip(x_values, original_surfaces[-1, :])):
                # smoothing_sample_range = delta_p
                if start <= x and x <= end and (x <= start + smoothing_sample_range or x >= end - smoothing_sample_range):
                    x_sample.append(x)
                    y_sample.append(y)
                    modified_indices.append(i)
            (approx, _, _) = points_approx(cheb_basis_shifted(start, end), [*x_sample], [*y_sample], max_degree=smoothing_degree)
            x_modified = x_values[modified_indices[0]:modified_indices[-1] + 1]
            y_modified = [approx(x) for x in x_modified]
            for i in range(modified_indices[0], modified_indices[-1] + 1):
                # if start + delta_p/2 <= x and x <= end - delta_p/2:
                    original_surfaces[-1, i] = y_modified[i - modified_indices[0]]

            if removal_epsilon > 0: # if removal epsilon is 0, do not remove points around where we start and (TODO: MAYBE?) end approximating
                i = 0
                while True:
                    if i >= len(x_values):
                        break
                    x = x_values[i]
                    if start - removal_epsilon <= x and x <= start + removal_epsilon:
                        x_values = np.delete(x_values, i)
                        original_surfaces = np.delete(original_surfaces, i, axis=1)
                    if end - removal_epsilon <= x and x <= end + removal_epsilon:
                        x_values = np.delete(x_values, i)
                        original_surfaces = np.delete(original_surfaces, i, axis=1)
                    i += 1

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

    # assert set(away_from_cusp_indices).union(set(near_cusp_indices)) == set(range(len(x_values)))

    esps_away = [elem_sym_poly(surface_count - 1, r) for r in range(1, surface_count)]
    esps_near = [elem_sym_poly(surface_count, r) for r in range(1, surface_count + 1)]
    x_values_away = [x_values[i] for i in away_from_cusp_indices]
    x_values_near = [x_values[i] for i in near_cusp_indices]
    esp_away_curves = [[esp(original_surfaces[0:-1, i].tolist()) for i in away_from_cusp_indices] for esp in esps_away]
    esp_near_curves = [[esp(original_surfaces[:, i].tolist()) for i in near_cusp_indices] for esp in esps_near]

    interpolated_curves_away = [polyval(x_values_away, polyfit(x_values_away, esp_curve, interpolant_degree)) for esp_curve in esp_away_curves] if len(x_values_away) > 0 else [] # The interpolated things are stand-ins for what a ML model would in practice give us
    interpolated_curves_near = [polyval(x_values_near, polyfit(x_values_near, esp_curve, interpolant_degree)) for esp_curve in esp_near_curves] if len(x_values_near) > 0 else [] # The interpolated things are stand-ins for what a ML model would in practice give us

    reconstructed_surfaces = np.empty(original_surfaces.shape)
    for i, values in enumerate(zip(*interpolated_curves_away)):
        roots = reconstruct_roots(values)
        idx = away_from_cusp_indices[i]
        reconstructed_surfaces[:, idx] = [*roots, math.nan] # include NaN at the end to make the lengths match
    for i, values in enumerate(zip(*interpolated_curves_near)):
        roots = reconstruct_roots(values)
        idx = near_cusp_indices[i]
        reconstructed_surfaces[:, idx] = roots

    return x_values, original_surfaces, reconstructed_surfaces # return x_values because it may have been modified

def reconstruct_roots(values):
    companion_column = [1] + [(-1)**(j + 1) * values[j] for j in range(len(values))]
    matrix = companion(companion_column) # scipy's Frobenius companion matrix implementation for simplicity
    roots = np.sort(np.linalg.eigvals(matrix))
    return roots

def reconstruct_frobenius(x_values, curves, interpolant_degree: int):
    original_surfaces = np.array([curves(x) for x in x_values]).T
    surface_count = original_surfaces.shape[0]
    esps = [elem_sym_poly(surface_count, r) for r in range(1, surface_count + 1)]
    esp_curves = [[esp(original_surfaces[:, i].tolist()) for i in range(original_surfaces.shape[1])] for esp in esps]
    interpolated_curves = [polyval(x_values, polyfit(x_values, esp_curve, interpolant_degree)) for esp_curve in esp_curves]
    reconstructed_surfaces = np.empty(original_surfaces.shape)
    for i, values in enumerate(zip(*interpolated_curves)):
        roots = reconstruct_roots(values)
        reconstructed_surfaces[:, i] = roots

    return original_surfaces, reconstructed_surfaces

def reconstruct_direct(x_values, curves, interpolant_degree: int):
    original_surfaces = np.array([curves(x) for x in x_values]).T
    reconstructed_surfaces = [polyval(x_values, polyfit(x_values, surface, interpolant_degree)) for surface in original_surfaces]
    return original_surfaces, np.array(reconstructed_surfaces)

samples = rand_unif_samples(0, 2, 4000)

def plot_reconstruction(x_values, original_surfaces, reconstructed_surfaces, name: str):
    plt.figure(0)
    plt.plot(x_values, original_surfaces[0, :])
    plt.plot(x_values, original_surfaces[1, :])
    plt.plot(x_values, original_surfaces[2, :])
    plt.title("Original multi-surface data")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

    plt.figure(1)
    plt.plot(x_values, reconstructed_surfaces[0, :])
    plt.plot(x_values, reconstructed_surfaces[1, :])
    plt.plot(x_values, reconstructed_surfaces[2, :])
    plt.title(f"ESP-reconstructed multi-surface ({name})")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

    errors = [np.abs(reconstructed_surfaces[i, :] - original_surfaces[i, :]) for i in range(reconstructed_surfaces.shape[0] - 1)]
    for error_points in errors:
        plt.plot(x_values, np.log10(error_points))
    plt.ylabel("log10 of error")
    plt.title(f"Error plot ({name})")
    plt.show()


def plot_params():
    values = np.linspace(0.01, 1, 100)
    max_abs_errors, gap_weighted_errors = [], []
    for value in values:
        _, original_surfaces, reconstructed_surfaces = reconstruct_enhanced(samples, curves=intersecting_cosine_curves_1, interpolant_degree=15, smoothing_degree=8, delta_p=value, matched_cusp_points=[np.pi/8, np.pi/6, np.pi/3, np.pi*5/8], removal_epsilon=0.0618, smoothing_sample_range=0.15)
        max_abs_errors.append(sum_max_abs_error(original_surfaces, reconstructed_surfaces))
        gap_weighted_errors.append(gap_weighted_error(original_surfaces, reconstructed_surfaces))

    plt.plot(values, max_abs_errors)
    plt.ylabel("max abs error")
    plt.xlabel("delta_p")
    plt.show()
    plt.plot(values, gap_weighted_errors)
    plt.ylabel("gap weighted error")
    plt.xlabel("delta_p")
    plt.show()
# plot_params()

x_values, original_surfaces, reconstructed_surfaces = reconstruct_enhanced(samples, curves=intersecting_cosine_curves_1, interpolant_degree=20, smoothing_degree=8, delta_p=0.211, matched_cusp_points=[np.pi/8, np.pi/6, np.pi/3, np.pi*5/8], removal_epsilon=0.618, smoothing_sample_range=0.15)
plot_reconstruction(x_values, original_surfaces, reconstructed_surfaces, name="Enhanced")

print("sum max abs error:", sum_max_abs_error(original_surfaces, reconstructed_surfaces))
print("gap weighted error:", gap_weighted_error(original_surfaces, reconstructed_surfaces))

original_surfaces, reconstructed_surfaces = reconstruct_frobenius(samples, curves=intersecting_cosine_curves_1, interpolant_degree=20)
plot_reconstruction(samples, original_surfaces, reconstructed_surfaces, name="Frobenius")

original_surfaces, reconstructed_surfaces = reconstruct_direct(samples, curves=intersecting_cosine_curves_1, interpolant_degree=20)
plot_reconstruction(samples, original_surfaces, reconstructed_surfaces, name="Direct")

def plot_convergence(error_fn, degrees, title: str):
    errors_enhanced = []
    errors_frobenius = []
    errors_direct = []
    for degree in degrees:
        _, original_surfaces, reconstructed_surfaces = reconstruct_enhanced(samples, curves=intersecting_cosine_curves_1, interpolant_degree=degree, smoothing_degree=8, delta_p=0.211, matched_cusp_points=[np.pi/8, np.pi/6, np.pi/3, np.pi*5/8], removal_epsilon=0.618, smoothing_sample_range=0.15)
        errors_enhanced.append(error_fn(original_surfaces, reconstructed_surfaces))

        original_surfaces, reconstructed_surfaces = reconstruct_frobenius(samples, curves=intersecting_cosine_curves_1, interpolant_degree=degree)
        errors_frobenius.append(error_fn(original_surfaces, reconstructed_surfaces))

        original_surfaces, reconstructed_surfaces = reconstruct_direct(samples, curves=intersecting_cosine_curves_1, interpolant_degree=degree)
        errors_direct.append(error_fn(original_surfaces, reconstructed_surfaces))
    
    plt.plot(degrees, errors_enhanced, label="Enhanced")
    plt.plot(degrees, errors_frobenius, label="Frobenius")
    plt.plot(degrees, errors_direct, label="Direct")
    plt.xlabel("Interpolant degree")
    plt.ylabel("Error")
    plt.title(title)
    plt.legend()
    plt.show()

# plot_convergence(gap_weighted_error, range(10, 45), title="Gap weighted error")
# plot_convergence(sum_max_abs_error, range(10, 45), title="Sum max abs error")
# plot_reconstruction(samples, original_surfaces, reconstructed_surfaces, name="Direct")

# compare with direct interpolation - put plots next to each other and error plots next to each other (or on same plot)
# plot convergence in degree (of interpolant) for direct interpolation, regular frob companion and this method
# think about proving error bounds (e.g. for derivatives)
# try to add a "request" for more data somewhere in algorithm to simulate real life
# try to make it work in 2d