from enum import unique
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import companion
import math
import random
from numpy.polynomial.chebyshev import chebfit as polyfit
from numpy.polynomial.chebyshev import chebval as polyval

from cusps.approx import points_approx
from cusps.errors import gap_weighted_error, sum_max_abs_error

np.random.seed(42) # make things reproducible

def intersecting_cosine_curves_1(x: float):
    y1 = math.sin(x)
    y2 = math.cos(2*x)
    y3 = math.sin(2*x)
    y4 = 1/3 + math.cos(2*x/3)
    return np.array(sorted([y1, y2, y3, y4])[0:3])

def chebyshev_t(n: int, x: float):
    return math.cos(n * math.acos(x))

def cheb_basis_shifted(a: float, b: float):
    return lambda n: lambda x: chebyshev_t(n, 2*(x - a)/(b - a) - 1)

def two_cheb_basis(n: int, m: int):
    return lambda x: chebyshev_t(n, x[0]) * chebyshev_t(m, x[1])

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

def reconstruct_roots(values):
    companion_column = [1] + [(-1)**(j + 1) * values[j] for j in range(len(values))]
    matrix = companion(companion_column) # scipy's Frobenius companion matrix implementation for simplicity
    roots = np.sort(np.linalg.eigvals(matrix))
    return roots

def estimate_matched_cusp_point(close_points): # given a set of sample points at which surfaces are close (potential matched cusp point candidates), estimate a representative for this set of points. weight by closeness
    return min(close_points, key=lambda x: x[1])[0]
    # return sum(point for (point, dist) in close_points) / len(close_points)

# TODO: write code for finding ubnmatched cusps using derivaives, adjust code for smoothing only there
def find_matched_cusp_points(sample_points, original_surfaces, interpolant_degree: int, tolerance: float, approx_basis, dimension: int):
    assert len(sample_points) == original_surfaces.shape[1]
    surface_count = original_surfaces.shape[0]
    esps = [elem_sym_poly(surface_count, r) for r in range(1, surface_count + 1)]
    esp_curves = [[esp(original_surfaces[:, i].tolist()) for i in range(original_surfaces.shape[1])] for esp in esps]
    approxes = [points_approx(approx_basis, sample_points, esp_curve, interpolant_degree, dimension) for esp_curve in esp_curves]
    interpolated_curves = [[approx[0](x) for x in sample_points] for approx in approxes]
    matched_cusp_points = []
    current_close_points = []
    for i, values in enumerate(zip(*interpolated_curves)):
        roots = reconstruct_roots(values)
        within_tolerance = False
        # break_outer = False
        # for j in range(len(roots)):
        #     for k in range(j + 1, len(roots)):
        #         dist = abs(roots[j] - roots[k])
        #         if dist < tolerance:
        #             within_tolerance = True
        #             current_close_points.append((sample_points[i], dist))
        #             break_outer = True
        #             break
        #     if break_outer:
        #         break
        dist = abs(roots[-2] - roots[-1])
        if dist < tolerance:
            within_tolerance = True
            current_close_points.append((sample_points[i], dist))
        if not within_tolerance and len(current_close_points) > 0:
            estimate = estimate_matched_cusp_point(current_close_points)
            matched_cusp_points.append(estimate)
            current_close_points = []
    if len(current_close_points) > 0:
        matched_cusp_points.append(estimate_matched_cusp_point(current_close_points))
    return matched_cusp_points

def unique_index_pairs(end: int):
    import itertools
    iterator = itertools.product(range(end), range(end))
    return filter(lambda i: i[0] > i[1], iterator)

matched_cusp_point_cache = []

def reconstruct_enhanced(a: float, b: float, sample_points, original_surfaces, interpolant_degree: int, smoothing_degree: int, delta_p: float, removal_epsilon: float, smoothing_sample_range: float, approx_basis, dimension: int, matched_cusp_point_search_tolerance: float):
    assert interpolant_degree > 0
    assert smoothing_degree >= 0
    assert delta_p >= 0
    assert removal_epsilon >= 0

    surface_count = original_surfaces.shape[0]

    global matched_cusp_point_cache
    matched_cusp_points = find_matched_cusp_points(sample_points, original_surfaces, interpolant_degree, tolerance=matched_cusp_point_search_tolerance, approx_basis=approx_basis, dimension=dimension) if matched_cusp_point_cache == [] else matched_cusp_point_cache
    matched_cusp_point_cache = matched_cusp_points
    # matched_cusp_points.pop(0)
    # matched_cusp_points = [np.pi/8, np.pi/3]
    # print("actual cusp points:", [np.pi/8, np.pi/3])
    # print("matched cusp points:", matched_cusp_points)

    if smoothing_degree > 0: # if smoothing degree is 0, do not attempt to smooth out the unmatched cusp
        for (start, end) in zip([a] + matched_cusp_points, matched_cusp_points + [b]):
            start = start
            end = end
            center = (start + end)/2
            radius = np.linalg.norm(end - start)/2
            x_sample = []
            y_sample = []
            modified_indices = []
            for i, (x, y) in enumerate(zip(sample_points, original_surfaces[-1, :])):
                # smoothing_sample_range = delta_p
                # if start <= x and x <= end and (x - start <= smoothing_sample_range or end - x <= smoothing_sample_range):
                if np.linalg.norm(x - center) <= radius and (np.linalg.norm(x - start) <= smoothing_sample_range or np.linalg.norm(x - end) <= smoothing_sample_range):
                    x_sample.append(x)
                    y_sample.append(y)
                    modified_indices.append(i)
            (approx, _, _) = points_approx(cheb_basis_shifted(start, end), x_sample, y_sample, max_degree=smoothing_degree, dimension=1) # NOTE: we can make the shift interval larger, and maintain the same accuracy. so can fit largest square around triangle in 2d
            x_modified = sample_points[modified_indices[0]:modified_indices[-1] + 1]
            y_modified = [approx(x) for x in x_modified]
            for i in range(modified_indices[0], modified_indices[-1] + 1):
                original_surfaces[-1, i] = y_modified[i - modified_indices[0]]
    if removal_epsilon > 0: # if removal epsilon is 0, do not remove points around where we approximate
        i = 0
        while True:
            if i >= len(sample_points):
                break
            x = sample_points[i]
            for p in matched_cusp_points:
                if np.linalg.norm(x - p) < removal_epsilon:
                    sample_points = np.delete(sample_points, i, axis=0)
                    original_surfaces = np.delete(original_surfaces, i, axis=1)
                    continue
            i += 1


    delta_q = delta_p + 0.01 # TODO: make this a parameter
    r = 1 # controls how much of a plateau at 1 we have. smaller r means a narrower plateau
    midpoints = [a] # TODO: rewrite for arbitrary dimensions
    for i in range(len(matched_cusp_points) - 1):
        midpoints.append((matched_cusp_points[i] + matched_cusp_points[i + 1])/2)
    midpoints.append(b)

    away_from_cusp_indices = []
    near_cusp_indices = []

    indices_full = []
    indices_bottom = []
    for i in range(len(sample_points)):
        if all(np.linalg.norm(sample_points[i] - p) >= delta_q for p in matched_cusp_points):
            indices_bottom.append(i)
        if any(np.linalg.norm(sample_points[i] - p) <= delta_p for p in matched_cusp_points):
            indices_full.append(i)
        near = False
        # if all(np.linalg.norm(sample_points[i] - p) >= delta_p for p in matched_cusp_points):
        #     indices_bottom.append(i)
        # full_index = True
        # x = sample_points[i]
        # for j in range(len(matched_cusp_points)):
        #     A = midpoints[j]
        #     C = matched_cusp_points[j] - delta_p
        #     B = (1 - r) * A + r*C
        #     D = matched_cusp_points[j] + delta_p
        #     F = midpoints[j + 1]
        #     E = (1 - r)*F + r*D
        #     if (A <= x and x <= B) or (E <= x and x <= F):
        #         full_index = False
        #         break
        # if full_index:
        #     indices_full.append(i)

        for p in matched_cusp_points:
            if np.linalg.norm(sample_points[i] - p) < delta_p:
                near_cusp_indices.append(i)
                near = True
                break
        if not near:
            away_from_cusp_indices.append(i)

    # print(indices_full)
    # assert set(indices_full).union(set(indices_bottom)) == set(range(len(sample_points)))
    # plt.scatter(indices_full, [0] * len(indices_full))
    # plt.scatter(indices_bottom, [1] * len(indices_bottom))
    # plt.show()
    def single_cusp_sigmoid(cusp, x):
        norm = np.linalg.norm(x - cusp)
        if norm >= delta_q:
            return 1
        if norm <= delta_p:
            return 0
        # r = delta_p / norm
        # start = (1 - r)*cusp + r*x
        # assert abs(np.linalg.norm(start - cusp) - delta_p) < 1e-8
        # s = delta_q / norm
        # end = (1 - s)*cusp + s*x
        # assert abs(np.linalg.norm(end - cusp) - delta_q) < 1e-8
        return sigmoid(delta_p, delta_q, norm)
    
    def product_sigmoid(x):
        return math.prod(single_cusp_sigmoid(cusp, x) for cusp in matched_cusp_points)
    
    def piecewise_sigmoid(x):
        for i in range(len(matched_cusp_points)):
            A = midpoints[i]
            C = matched_cusp_points[i] - delta_p
            B = (1 - r) * A + r*C
            D = matched_cusp_points[i] + delta_p
            F = midpoints[i + 1]
            E = (1 - r)*F + r*D
            if A <= x and x <= B:
                return 1
            if B <= x and x <= C:
                return 1 - sigmoid(B, C, x)
            if C <= x and x <= D:
                return 0
            if D <= x and x <= E:
                return sigmoid(D, E, x)
            if E <= x and x <= F:
                return 1
        raise Exception("piecewise_sigmoid range error")
    
    # plt.plot(sample_points, [piecewise_sigmoid(x) for x in sample_points])
    # plt.plot(sample_points, [product_sigmoid(x) for x in sample_points])
    # plt.show()
    
    esps_away = [elem_sym_poly(surface_count - 1, r) for r in range(1, surface_count)]
    esps_near = [elem_sym_poly(surface_count, r) for r in range(1, surface_count + 1)]

    sample_points_away = [sample_points[i] for i in away_from_cusp_indices]
    sample_points_near = [sample_points[i] for i in near_cusp_indices]
    esp_away_curves = [[esp(original_surfaces[0:-1, i].tolist()) for i in away_from_cusp_indices] for esp in esps_away]
    esp_near_curves = [[esp(original_surfaces[:, i].tolist()) for i in near_cusp_indices] for esp in esps_near]
    approxes_away = [points_approx(approx_basis, sample_points_away, esp_curve, interpolant_degree, dimension) for esp_curve in esp_away_curves] if len(sample_points_away) > 0 else []
    approxes_near = [points_approx(approx_basis, sample_points_near, esp_curve, interpolant_degree, dimension) for esp_curve in esp_near_curves] if len(sample_points_near) > 0 else []
    interpolated_curves_away = [[approx[0](x) for x in sample_points_away] for approx in approxes_away]
    interpolated_curves_near = [[approx[0](x) for x in sample_points_near] for approx in approxes_near] # TODO: for some reason, this gives better errors than using polyfit and polyval

    sample_points_full = [sample_points[i] for i in indices_full]
    sample_points_bottom = [sample_points[i] for i in indices_bottom]
    esp_curves_full = [[esp(original_surfaces[:, i].tolist()) for i in indices_full] for esp in esps_near]
    esp_curves_bottom = [[esp(original_surfaces[0:-1, i].tolist()) for i in indices_bottom] for esp in esps_away]
    approxes_full = [points_approx(approx_basis, sample_points_full, esp_curve, interpolant_degree, dimension) for esp_curve in esp_curves_full] if len(sample_points_full) > 0 else []
    approxes_bottom = [points_approx(approx_basis, sample_points_bottom, esp_curve, interpolant_degree, dimension) for esp_curve in esp_curves_bottom] if len(sample_points_bottom) > 0 else []
    interpolated_curves_full = [[approx[0](x) for x in sample_points] for approx in approxes_full] # NOTE: for some reason, this gives better errors than using polyfit and polyval
    interpolated_curves_bottom = [[approx[0](x) for x in sample_points] for approx in approxes_bottom] # NOTE: for some reason, this gives better errors than using polyfit and polyval

    reconstructed_surfaces = np.empty(original_surfaces.shape)
    # for i, values in enumerate(zip(*interpolated_curves_away)):
    #     roots = reconstruct_roots(values)
    #     idx = away_from_cusp_indices[i]
    #     reconstructed_surfaces[:, idx] = [*roots, math.nan] # include NaN at the end to make the lengths match
    # for i, values in enumerate(zip(*interpolated_curves_near)):
    #     roots = reconstruct_roots(values)
    #     idx = near_cusp_indices[i]
    #     reconstructed_surfaces[:, idx] = roots
    for i, values in enumerate(zip(*interpolated_curves_full)):
        roots = reconstruct_roots(values)
        if len(sample_points_bottom) == 0:
            reconstructed_surfaces[:, i] = roots
        else:
            values_bottom = np.array(interpolated_curves_bottom)[:, i]
            roots_bottom = np.array([*reconstruct_roots(values_bottom), math.nan])
            x = sample_points[i]
            alpha = product_sigmoid(x)
            reconstructed_surfaces[:, i] = alpha * roots_bottom + (1 - alpha) * roots

    return sample_points, original_surfaces, reconstructed_surfaces # return sample_points and original_surfaces because they may have been modified

def sigmoid(a, b, x):
    def f(x):
        if x > 0:
            return math.exp(-1/x)
        return 0
    def g(x):
        return f(x) / (f(x) + f(1 - x))
    return g((x - a)/(b - a))

def reconstruct_frobenius(sample_points, curves, interpolant_degree: int, approx_basis, dimension: int):
    original_surfaces = np.array([curves(x) for x in sample_points]).T
    surface_count = original_surfaces.shape[0]
    esps = [elem_sym_poly(surface_count, r) for r in range(1, surface_count + 1)]
    esp_curves = [[esp(original_surfaces[:, i].tolist()) for i in range(original_surfaces.shape[1])] for esp in esps]
    approxes = [points_approx(approx_basis, sample_points, esp_curve, interpolant_degree, dimension) for esp_curve in esp_curves]
    interpolated_curves = [[approx[0](x) for x in sample_points] for approx in approxes]
    # interpolated_curves = [polyval(sample_points, polyfit(sample_points, esp_curve, interpolant_degree)) for esp_curve in esp_curves]
    
    reconstructed_surfaces = np.empty(original_surfaces.shape)
    for i, values in enumerate(zip(*interpolated_curves)):
        roots = reconstruct_roots(values)
        reconstructed_surfaces[:, i] = roots

    return original_surfaces, reconstructed_surfaces

def reconstruct_direct(sample_points, curves, interpolant_degree: int, approx_basis, dimension: int):
    original_surfaces = np.array([curves(x) for x in sample_points]).T
    approxes = [points_approx(approx_basis, sample_points, surface, interpolant_degree, dimension) for surface in original_surfaces]
    reconstructed_surfaces = [[approx[0](x) for x in sample_points] for approx in approxes]
    # reconstructed_surfaces = [polyval(sample_points, polyfit(sample_points, surface, interpolant_degree)) for surface in original_surfaces]
    return original_surfaces, np.array(reconstructed_surfaces)