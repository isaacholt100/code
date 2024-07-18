from typing import TypedDict
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import Delaunay

from cusps.approx import points_approx
from cusps.esps import elem_sym_poly, reconstruct_roots
from cusps.find_cusps import find_matched_cusp_points
from cusps.transition import matched_cusps_approx_weight, simplex_approx_weight

np.random.seed(42) # make things reproducible

def chebyshev_t(n: int, x: float) -> float:
    return math.cos(n * math.acos(x))

def cheb_basis_shifted(a: float, b: float):
    return lambda n: lambda x: chebyshev_t(n, 2*(x - a)/(b - a) - 1)

def two_cheb_basis(n: int, m: int):
    return lambda x: chebyshev_t(n, x[0]) * chebyshev_t(m, x[1])

class MatchCuspParams(TypedDict):
    value_tolerance: float
    point_tolerance: float

def hypercuboid_vertices(sides: list[tuple[float, float]]) -> list[list[float]]:
    assert len(sides) > 0
    if len(sides) == 1:
        return [[sides[0][0]], [sides[0][1]]]
    vertices_lower = hypercuboid_vertices(sides[1:])
    vertices = []
    for vertex in vertices_lower:
        assert len(vertex) == len(sides) - 1
        vertices.append([sides[0][0]] + vertex.copy())
        vertices.append([sides[0][1]] + vertex)
    
    assert len(vertices) == 2**len(sides)
    return vertices

def reconstruct_enhanced(domains: list[tuple[float, float]], sample_points, original_surfaces, interpolant_degree: int, delta_p: float, delta_q: float, removal_epsilon: float, approx_basis, match_cusps: MatchCuspParams | list, use_delaunay = False):
    assert interpolant_degree > 0
    assert delta_p >= 0
    assert delta_q >= delta_p
    assert removal_epsilon >= 0

    dimension = len(domains)
    assert dimension > 0

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
    for i in range(len(sample_points)):
        if all(np.linalg.norm(sample_points[i] - p) >= delta_q for p in matched_cusp_points):
            indices_bottom.append(i)
        if any(np.linalg.norm(sample_points[i] - p) <= delta_p for p in matched_cusp_points):
            indices_full.append(i)
    
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

    reconstructed_surfaces = np.empty(original_surfaces.shape)
    if use_delaunay:
        triangulation_points = matched_cusp_points + hypercuboid_vertices(domains)
        triangulation = Delaunay(triangulation_points)
        simplices = triangulation.simplices

        indices_delaunay = [[] for _ in range(len(simplices))]
        for i in indices_bottom:
            simplex_index = triangulation.find_simplex(sample_points[i])
            if simplex_index != -1:
                indices_delaunay[simplex_index].append(i)
        for index in indices_delaunay:
            assert len(index) > 0 # this is because the simplices triangulate the entire domain

        sample_points_delaunay = [[sample_points[i] for i in idx] for idx in indices_delaunay]

        esp_curves_delaunay = [[[esp(original_surfaces[0:-1, i].tolist()) for i in idx] for esp in esps_bottom] for idx in indices_delaunay]

        approxes_delaunay = [[points_approx(approx_basis, sample_points_delaunay[j], esp_curve, interpolant_degree, dimension) for esp_curve in esp_curves_delaunay[j]] for j in range(len(simplices))]

        interpolated_curves_delaunay = [[[approx[0](x) for x in sample_points] for approx in approxes] for approxes in approxes_delaunay]
        
        plt_fn = lambda: plt.triplot(np.array(triangulation_points)[:, 0], np.array(triangulation_points)[:, 1], triangulation.simplices, color="red")

        for i in range(len(sample_points)):
            simplex_index = triangulation.find_simplex(sample_points[i])
            assert simplex_index != -1
            weights = [simplex_approx_weight(delta_p, delta_q, triangulation_points, triangulation, sample_points[i], j) for j in range(len(simplices))]
            alpha = matched_cusps_approx_weight(delta_p, delta_q, matched_cusp_points, sample_points[i])

            values_unmerged = [np.array(interpolated_curves_delaunay[j])[:, i] for j in range(len(simplices))]
            roots_full = reconstruct_roots(np.array(interpolated_curves_full)[:, i])
            roots_unmerged = [np.array([*reconstruct_roots(values_unmerged[j]) * weights[j], math.nan]) for j in range(len(simplices))] + [roots_full * (1 - alpha)]
            roots = np.sum(roots_unmerged, axis=0) / (sum(weights) + (1 - alpha))
            reconstructed_surfaces[:, i] = roots
            # values_unmerged = np.vstack(([np.hstack((np.array(interpolated_curves_delaunay[j])[:, i], [0])) * weights[j] for j in range(len(simplices))], np.array(interpolated_curves_full)[:, i] * (1 - alpha)))
            # values = np.sum(values_unmerged, axis=0) / (sum(weights) + (1 - alpha))
            # # roots_delaunay = np.array([*reconstruct_roots(values_delaunay), math.nan])
            # # roots_full = reconstruct_roots(np.array(interpolated_curves_full)[:, i])
            # # roots_unmerged = [np.array([*reconstruct_roots(values_unmerged[j]) * weights[j], math.nan]) for j in range(len(simplices))] + [roots_full * (1 - alpha)]
            # # roots = np.sum(roots_unmerged, axis=0) / (sum(weights) + (1 - alpha))
            # # roots = alpha * roots_delaunay + (1 - alpha)*roots_full
            # if np.any(np.isnan(values)):
            #     reconstructed_surfaces[:, i] = [math.nan] * len(values) # TODO: find a way of handling/removing this condition
            # roots = reconstruct_roots(values).tolist()
            # if alpha == 1:
            #     # remove zero from roots, since this corresponds to a fake value of the top surface (which we don't care about)
            #     idx = roots.index(0)
            #     roots.pop(idx)
            #     roots.append(math.nan)
            # reconstructed_surfaces[:, i] = roots

        ## TODO: instead of weighting the roots, could try weighting the values?

        return sample_points, original_surfaces, reconstructed_surfaces, plt_fn # return sample_points and original_surfaces because they may have been modified
    
    for i in range(len(sample_points)):
        if len(sample_points_bottom) == 0:
            roots = reconstruct_roots(np.array(interpolated_curves_full)[:, i])
            reconstructed_surfaces[:, i] = roots
        elif len(sample_points_full) == 0:
            values_bottom = np.array(interpolated_curves_bottom)[:, i]
            roots_bottom = np.array([*reconstruct_roots(values_bottom), math.nan])
            reconstructed_surfaces[:, i] = roots_bottom
        else:
            x = sample_points[i]
            alpha = matched_cusps_approx_weight(delta_p, delta_q, matched_cusp_points, x)
            # roots = reconstruct_roots(np.array(interpolated_curves_full)[:, i] * (1 - alpha) + np.hstack((np.array(interpolated_curves_bottom)[:, i], [0])) * alpha).tolist()
            # if alpha == 1:
            #     # remove zero from roots, since this corresponds to a fake value of the top surface (which we don't care about)
            #     idx = roots.index(0)
            #     roots.pop(idx)
            #     roots.append(math.nan)
            # if x == 1.3054922157037494:
            #     print(roots[1], x, alpha)
            #     roots[1] = 0.490
            # reconstructed_surfaces[:, i] = np.array(roots)
            # continue
            roots = reconstruct_roots(np.array(interpolated_curves_full)[:, i])
            values_bottom = np.array(interpolated_curves_bottom)[:, i]
            roots_bottom = np.array([*reconstruct_roots(values_bottom), math.nan])
            reconstructed_surfaces[:, i] = alpha * roots_bottom + (1 - alpha) * roots

    return sample_points, original_surfaces, reconstructed_surfaces, lambda: () # return sample_points and original_surfaces because they may have been modified

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

## TODO: add derivative plot for 1D