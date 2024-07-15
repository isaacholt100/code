from typing import Literal, Tuple, TypedDict
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import companion
import math
from scipy.spatial import Delaunay

from cusps.approx import points_approx

np.random.seed(42) # make things reproducible

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

def estimate_matched_cusp_point(close_points): # given a set of sample points at which surfaces are close (potential matched cusp point candidates), estimate a representative for this set of points
    return min(close_points, key=lambda x: x[1])[0]

# TODO: write code for finding ubnmatched cusps using derivaives, adjust code for smoothing only there

def find_matched_cusp_points_old(sample_points, original_surfaces, interpolant_degree: int, value_tolerance: float, approx_basis, dimension: int):
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
        dist = abs(roots[-2] - roots[-1])
        if dist < value_tolerance:
            within_tolerance = True
            current_close_points.append((sample_points[i], dist))
        if not within_tolerance and len(current_close_points) > 0:
            estimate = estimate_matched_cusp_point(current_close_points)
            matched_cusp_points.append(estimate)
            current_close_points = []
    if len(current_close_points) > 0:
        matched_cusp_points.append(estimate_matched_cusp_point(current_close_points))
    return matched_cusp_points

def cluster_and_estimate(close_points: list[Tuple], point_tolerance: float):
    close_points.sort(key=lambda x: x[1])
    matched_cusp_points = []
    i = 0
    while True:
        if i >= len(close_points):
            break
        current_cluster = [close_points[i]]
        j = i + 1
        while True:
            if j >= len(close_points):
                break
            if np.linalg.norm(close_points[i][0] - close_points[j][0]) < point_tolerance:
                p = close_points.pop(j)
                current_cluster.append(p)
            else:
                j += 1
        matched_cusp_points.append(current_cluster[0][0]) # since we sorted the array by distance at the start of the function, the first item of the current cluster is guaranteed to be the one with minimal distance
        i += 1
    return matched_cusp_points

def find_matched_cusp_points(sample_points, original_surfaces, interpolant_degree: int, value_tolerance: float, point_tolerance: float, approx_basis, dimension: int):
    assert len(sample_points) == original_surfaces.shape[1]
    surface_count = original_surfaces.shape[0]
    esps = [elem_sym_poly(surface_count, r) for r in range(1, surface_count + 1)]
    esp_curves = [[esp(original_surfaces[:, i].tolist()) for i in range(original_surfaces.shape[1])] for esp in esps]
    approxes = [points_approx(approx_basis, sample_points, esp_curve, interpolant_degree, dimension) for esp_curve in esp_curves]
    interpolated_curves = [[approx[0](x) for x in sample_points] for approx in approxes]
    close_points = []
    for i, values in enumerate(zip(*interpolated_curves)):
        roots = reconstruct_roots(values)
        dist = abs(roots[-2] - roots[-1])
        if dist < value_tolerance:
            close_points.append((sample_points[i], dist))
    # print(close_points)
    matched_cusp_points = cluster_and_estimate(close_points, point_tolerance)
    return matched_cusp_points

def unique_index_pairs(end: int):
    import itertools
    iterator = itertools.product(range(end), range(end))
    return filter(lambda i: i[0] > i[1], iterator)

class MatchCuspParams(TypedDict):
    value_tolerance: float
    point_tolerance: float

def reconstruct_enhanced(sample_points, original_surfaces, interpolant_degree: int, delta_p: float, delta_q: float, removal_epsilon: float, approx_basis, dimension: int, match_cusps: MatchCuspParams | list, use_delaunay = False):
    assert interpolant_degree > 0
    assert delta_p >= 0
    assert delta_q >= delta_p
    assert removal_epsilon >= 0

    surface_count = original_surfaces.shape[0]

    matched_cusp_points = find_matched_cusp_points(sample_points, original_surfaces, interpolant_degree, value_tolerance=match_cusps["value_tolerance"], point_tolerance=match_cusps["point_tolerance"], approx_basis=approx_basis, dimension=dimension) if isinstance(match_cusps, dict) else match_cusps

    if removal_epsilon > 0:
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

    indices_full = []
    indices_bottom = []
    # if use_delaunay:
    #     delta_q = delta_p # TODO: change later, this is just for testing Delaunay
    for i in range(len(sample_points)):
        if all(np.linalg.norm(sample_points[i] - p) >= delta_q for p in matched_cusp_points):
            indices_bottom.append(i)
        if any(np.linalg.norm(sample_points[i] - p) <= delta_p for p in matched_cusp_points):
            indices_full.append(i)

    def single_cusp_sigmoid(cusp, x):
        norm = np.linalg.norm(x - cusp)
        if norm >= delta_q:
            return 1
        if norm <= delta_p:
            return 0
        return sigmoid(delta_p, delta_q, norm)
    
    def product_sigmoid(x):
        return math.prod(single_cusp_sigmoid(cusp, x) for cusp in matched_cusp_points)
    
    esps_full = [elem_sym_poly(surface_count, r) for r in range(1, surface_count + 1)]
    esps_bottom = [elem_sym_poly(surface_count - 1, r) for r in range(1, surface_count)]

    sample_points_full = [sample_points[i] for i in indices_full]
    sample_points_bottom = [sample_points[i] for i in indices_bottom]

    esp_curves_full = [[esp(original_surfaces[:, i].tolist()) for i in indices_full] for esp in esps_full]
    esp_curves_bottom = [[esp(original_surfaces[0:-1, i].tolist()) for i in indices_bottom] for esp in esps_bottom]

    approxes_full = [points_approx(approx_basis, sample_points_full, esp_curve, interpolant_degree, dimension) for esp_curve in esp_curves_full] if len(sample_points_full) > 0 else []
    approxes_bottom = [points_approx(approx_basis, sample_points_bottom, esp_curve, interpolant_degree, dimension) for esp_curve in esp_curves_bottom] if len(sample_points_bottom) > 0 else []

    interpolated_curves_full = [[approx[0](x) for x in sample_points] for approx in approxes_full]
    interpolated_curves_bottom = [[approx[0](x) for x in sample_points] for approx in approxes_bottom]

    if use_delaunay:
        triangulation_points = np.array(matched_cusp_points + [[-1, -1], [-1, 1], [1, -1], [1, 1]]) # TODO: make this generalisable to arbitrary domains (domain needs to be tensor product of intervals)
        triangulation = Delaunay(triangulation_points)
        simplices = triangulation.simplices

        indices_delaunay = [[] for _ in range(len(simplices))]
        new_indices_bottom = []
        indices_delaunay_all = []
        for i in indices_bottom:
            simplex_index = triangulation.find_simplex(sample_points[i])
            if simplex_index != -1:
                indices_delaunay[simplex_index].append(i)
                indices_delaunay_all.append(i)
            else:
                new_indices_bottom.append(i)

        indices_bottom = new_indices_bottom

        sample_points_delaunay = [[sample_points[i] for i in idx] for idx in indices_delaunay]
        sample_points_delaunay_all = [sample_points[i] for i in indices_delaunay_all]

        esp_curves_delaunay = [[[esp(original_surfaces[0:-1, i].tolist()) for i in idx] for esp in esps_bottom] for idx in indices_delaunay]
        esp_curves_delaunay_all = [[esp(original_surfaces[0:-1, i].tolist()) for i in indices_delaunay_all] for esp in esps_bottom]

        approxes_delaunay = [[points_approx(approx_basis, sample_points_delaunay[j], esp_curve, interpolant_degree, dimension) for esp_curve in esp_curves_delaunay[j]] if len(sample_points_delaunay[j]) > 0 else [] for j in range(len(simplices))]
        approxes_delaunay_all = [points_approx(approx_basis, sample_points_delaunay_all, esp_curve, interpolant_degree, dimension) for esp_curve in esp_curves_delaunay_all] if len(sample_points_delaunay_all) > 0 else []

        interpolated_curves_delaunay = [[[approx[0](x) for x in sample_points] for approx in approxes] for approxes in approxes_delaunay]
        interpolated_curves_delaunay_all = [[approx[0](x) for x in sample_points] for approx in approxes_delaunay_all]

    # for esp in esps_bottom:
    #     break
    #     # plt.plot(sample_points, [esp(original_surfaces[0:-1, i].tolist()) for i in range(len(sample_points))])
    # plt.plot(sample_points, interpolated_curves_bottom)
    # for curve in interpolated_curves_bottom:
    #     plt.plot(sample_points, curve)
    # plt.show()

    # for esp in esps_full:
    #     break
    #     # plt.plot(sample_points, [esp(original_surfaces[:, i].tolist()) for i in range(len(sample_points))])
    # for curve in interpolated_curves_full:
    #     plt.plot(sample_points, curve)
    # plt.show()

        def barycentric_coords(x, x1, x2, x3):
            det = (x2[1] - x3[1])*(x1[0] - x3[0]) + (x3[0] - x2[0])*(x1[1] - x3[1])
            lambda1 = ((x2[1] - x3[1])*(x[0] - x3[0]) + (x3[0] - x2[0])*(x[1] - x3[1])) / det
            lambda2 = ((x3[1] - x1[1])*(x[0] - x3[0]) + (x1[0] - x3[0])*(x[1] - x3[1])) / det
            lambda3 = 1 - lambda1 - lambda2
            return [lambda1, lambda2, lambda3]
        
        def transition_component(x, simplex_index: int):
            simplex_vertices = [triangulation_points[i] for i in triangulation.simplices[simplex_index]]
            return math.prod(smooth_step(l) for l in barycentric_coords(x, simplex_vertices[0], simplex_vertices[1], simplex_vertices[2])) * math.prod(single_cusp_sigmoid(vertex, x) for vertex in simplex_vertices)

        def f(x):
            if x > 0:
                return math.exp(-1/x)
            return 0

        def g(x):
            return f(x) / (f(x) + f(1 - x))

        def smooth_step(x):
            a = 0
            b = 0.01
            return g((x - a)/(b - a))
        
        plt_fn = lambda: plt.triplot(triangulation_points[:, 0], triangulation_points[:, 1], triangulation.simplices, color="red")
        # plt.show()

        reconstructed_surfaces = np.empty(original_surfaces.shape)
        for i in range(len(sample_points)):
            # if i in indices_full:
            #     roots = reconstruct_roots(np.array(interpolated_curves_full)[:, i])
            #     reconstructed_surfaces[:, i] = roots
            #     continue
            simplex_index = triangulation.find_simplex(sample_points[i])
            # assert simplex_index != -1
            if True or simplex_index != -1:
                weights = [transition_component(sample_points[i], j) for j in range(len(simplices))]
                weights_test = np.zeros(len(simplices))
                weights_test[simplex_index] = 1
                alpha = product_sigmoid(sample_points[i])

                values_unmerged = [np.array(interpolated_curves_delaunay[j])[:, i] for j in range(len(simplices))]
                roots_full = reconstruct_roots(np.array(interpolated_curves_full)[:, i])
                roots_unmerged = [np.array([*reconstruct_roots(values_unmerged[j]) * weights[j], math.nan]) for j in range(len(simplices))] + [roots_full * (1 - alpha)]
                roots = np.sum(roots_unmerged, axis=0) / (np.sum(weights) + (1 - alpha))
                reconstructed_surfaces[:, i] = roots
                continue
            if simplex_index != -1 and len(sample_points_delaunay[simplex_index]) > 0:
                values = np.array(interpolated_curves_delaunay[simplex_index])[:, i]
                roots = np.array([*reconstruct_roots(values), math.nan])
                reconstructed_surfaces[:, i] = roots
                continue
            if len(sample_points_bottom) == 0:
                roots = reconstruct_roots(np.array(interpolated_curves_full)[:, i])
                reconstructed_surfaces[:, i] = roots
            elif len(sample_points_full) == 0:
                values_bottom = np.array(interpolated_curves_bottom)[:, i]
                roots_bottom = np.array([*reconstruct_roots(values_bottom), math.nan])
                reconstructed_surfaces[:, i] = roots_bottom
            else:
                roots = reconstruct_roots(np.array(interpolated_curves_full)[:, i])
                values_bottom = np.array(interpolated_curves_bottom)[:, i]
                roots_bottom = np.array([*reconstruct_roots(values_bottom), math.nan])
                x = sample_points[i]
                alpha = product_sigmoid(x)
                reconstructed_surfaces[:, i] = alpha * roots_bottom + (1 - alpha) * roots

        return sample_points, original_surfaces, reconstructed_surfaces, plt_fn # return sample_points and original_surfaces because they may have been modified
    reconstructed_surfaces = np.empty(original_surfaces.shape)
    for i in range(len(sample_points)):
        if len(sample_points_bottom) == 0:
            roots = reconstruct_roots(np.array(interpolated_curves_full)[:, i])
            reconstructed_surfaces[:, i] = roots
        elif len(sample_points_full) == 0:
            values_bottom = np.array(interpolated_curves_bottom)[:, i]
            roots_bottom = np.array([*reconstruct_roots(values_bottom), math.nan])
            reconstructed_surfaces[:, i] = roots_bottom
        else:
            roots = reconstruct_roots(np.array(interpolated_curves_full)[:, i])
            values_bottom = np.array(interpolated_curves_bottom)[:, i]
            roots_bottom = np.array([*reconstruct_roots(values_bottom), math.nan])
            x = sample_points[i]
            alpha = product_sigmoid(x)
            reconstructed_surfaces[:, i] = alpha * roots_bottom + (1 - alpha) * roots

    return sample_points, original_surfaces, reconstructed_surfaces, lambda: () # return sample_points and original_surfaces because they may have been modified

def reconstruct_enhanced_1d(a: float, b: float, sample_points, original_surfaces, interpolant_degree: int, smoothing_degree: int, delta_p: float, removal_epsilon: float, smoothing_sample_range: float, approx_basis, dimension: int, matched_cusp_point_search_tolerance: float):
    assert interpolant_degree > 0
    assert smoothing_degree >= 0
    assert delta_p >= 0
    assert removal_epsilon >= 0

    surface_count = original_surfaces.shape[0]

    matched_cusp_points = find_matched_cusp_points(sample_points, original_surfaces, interpolant_degree, value_tolerance=matched_cusp_point_search_tolerance, point_tolerance=0.1, approx_basis=approx_basis, dimension=dimension) # TODO: add param for point tolerance, if we're even still using this function
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

    delta_q = delta_p + 0.01

    indices_full = []
    indices_bottom = []
    for i in range(len(sample_points)):
        if all(np.linalg.norm(sample_points[i] - p) >= delta_q for p in matched_cusp_points):
            indices_bottom.append(i)
        if any(np.linalg.norm(sample_points[i] - p) <= delta_p for p in matched_cusp_points):
            indices_full.append(i)

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
        return sigmoid(delta_p, delta_q, norm)
    
    def product_sigmoid(x):
        return math.prod(single_cusp_sigmoid(cusp, x) for cusp in matched_cusp_points)
    
    esps_full = [elem_sym_poly(surface_count - 1, r) for r in range(1, surface_count)]
    esps_bottom = [elem_sym_poly(surface_count, r) for r in range(1, surface_count + 1)]

    sample_points_full = [sample_points[i] for i in indices_full]
    sample_points_bottom = [sample_points[i] for i in indices_bottom]
    esp_curves_full = [[esp(original_surfaces[:, i].tolist()) for i in indices_full] for esp in esps_bottom]
    esp_curves_bottom = [[esp(original_surfaces[0:-1, i].tolist()) for i in indices_bottom] for esp in esps_full]
    approxes_full = [points_approx(approx_basis, sample_points_full, esp_curve, interpolant_degree, dimension) for esp_curve in esp_curves_full] if len(sample_points_full) > 0 else []
    approxes_bottom = [points_approx(approx_basis, sample_points_bottom, esp_curve, interpolant_degree, dimension) for esp_curve in esp_curves_bottom] if len(sample_points_bottom) > 0 else []
    interpolated_curves_full = [[approx[0](x) for x in sample_points] for approx in approxes_full] # NOTE: for some reason, this gives better errors than using polyfit and polyval
    interpolated_curves_bottom = [[approx[0](x) for x in sample_points] for approx in approxes_bottom] # NOTE: for some reason, this gives better errors than using polyfit and polyval

    reconstructed_surfaces = np.empty(original_surfaces.shape)
    for i, (values_full, values_bottom) in enumerate(zip(zip(*interpolated_curves_full), zip(*interpolated_curves_bottom))):
        roots_full = reconstruct_roots(values_full)
        if len(sample_points_bottom) == 0:
            reconstructed_surfaces[:, i] = roots_full
        else:
            roots_bottom = np.array([*reconstruct_roots(values_bottom), math.nan])
            x = sample_points[i]
            alpha = product_sigmoid(x)
            reconstructed_surfaces[:, i] = alpha * roots_bottom + (1 - alpha) * roots_full

    return sample_points, original_surfaces, reconstructed_surfaces # return sample_points and original_surfaces because they may have been modified

def sigmoid(a, b, x):
    def f(x):
        if x > 0:
            return math.exp(-1/x)
        return 0
    def g(x):
        return f(x) / (f(x) + f(1 - x))
    return g((x - a)/(b - a))

def reconstruct_frobenius(sample_points, original_surfaces, interpolant_degree: int, approx_basis, dimension: int):
    surface_count = original_surfaces.shape[0]
    esps = [elem_sym_poly(surface_count, r) for r in range(1, surface_count + 1)]
    esp_curves = [[esp(original_surfaces[:, i].tolist()) for i in range(original_surfaces.shape[1])] for esp in esps]
    approxes = [points_approx(approx_basis, sample_points, esp_curve, interpolant_degree, dimension) for esp_curve in esp_curves]
    interpolated_curves = [[approx[0](x) for x in sample_points] for approx in approxes]
    
    reconstructed_surfaces = np.empty(original_surfaces.shape)
    for i, values in enumerate(zip(*interpolated_curves)):
        roots = reconstruct_roots(values)
        reconstructed_surfaces[:, i] = roots

    return original_surfaces, reconstructed_surfaces

def reconstruct_direct(sample_points, curves, interpolant_degree: int, approx_basis, dimension: int):
    original_surfaces = np.array([curves(x) for x in sample_points]).T
    approxes = [points_approx(approx_basis, sample_points, surface, interpolant_degree, dimension) for surface in original_surfaces]
    reconstructed_surfaces = [[approx[0](x) for x in sample_points] for approx in approxes]
    return original_surfaces, np.array(reconstructed_surfaces)