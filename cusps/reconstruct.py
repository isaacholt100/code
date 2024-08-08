from typing import Any, TypedDict
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import Delaunay

from bases import Basis, BasisFunction, chebyshev_t
from cusps.approx import points_approx
from cusps.esps import elem_sym_poly, reconstruct_roots
from cusps.find_cusps import MatchCuspParams, find_matched_cusp_points, reconstruct_frobenius
from cusps.transition import matched_cusps_approx_weight, simplex_approx_weight
from custom_types import Curves, FloatArray

def deriv(y_values, delta_x):
    derivs = [(y_values[i + 1] - y_values[i - 1])/(2*delta_x) for i in range(1, len(y_values) - 1)]
    return derivs

np.random.seed(42) # make things reproducible

def cheb_basis_shifted(a: float, b: float) -> Basis:
    return lambda idx: lambda x: chebyshev_t(idx[0], 2*(x[0] - a)/(b - a) - 1)

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

def get_surface_values(curves: Curves, samples: FloatArray) -> FloatArray:
    return np.array([curves(*x) for x in samples])

def reconstruct_enhanced(
    domains: list[tuple[float, float]],
    sample_points: FloatArray,
    approximation_points: FloatArray,
    original_surface_values: FloatArray,
    interpolant_degree: int,
    delta_p: float,
    delta_q: float,
    removal_epsilon: float,
    approx_basis: Basis,
    match_cusps: MatchCuspParams | list[FloatArray],
    use_delaunay = False
) -> tuple[FloatArray, Any]:
    assert sample_points.ndim == 2
    assert approximation_points.ndim == 2
    assert original_surface_values.ndim == 2
    assert sample_points.shape[0] == original_surface_values.shape[0]
    assert sample_points.shape[1] == approximation_points.shape[1]
    assert sample_points.shape[1] == len(domains)

    assert interpolant_degree > 0
    assert delta_p >= 0
    assert delta_q >= delta_p
    assert removal_epsilon >= 0

    dimension = len(domains)
    assert dimension > 0
    surface_count = original_surface_values.shape[1]

    reconstructed_frob = reconstruct_frobenius(
        sample_points,
        approximation_points,
        original_surface_values,
        interpolant_degree,
        approx_basis
    )

    matched_cusp_points = find_matched_cusp_points(approximation_points, reconstructed_frob, match_cusps) if isinstance(match_cusps, dict) else match_cusps
    print("matched cusp points:", matched_cusp_points)

    if removal_epsilon > 0:
        i = 0
        while True:
            if i >= len(sample_points):
                break
            x = sample_points[i]
            for p in matched_cusp_points:
                if np.linalg.norm(x - p) < removal_epsilon:
                    sample_points = np.delete(sample_points, i, axis=0)
                    original_surface_values = np.delete(original_surface_values, i, axis=0)
                    continue
            i += 1

    indices_full = []
    indices_bottom = []
    for i in range(len(sample_points)):
        if all(np.linalg.norm(sample_points[i] - p) >= delta_p for p in matched_cusp_points):
            indices_bottom.append(i)
        if any(np.linalg.norm(sample_points[i] - p) <= delta_q for p in matched_cusp_points):
            indices_full.append(i)

    sample_points_full = sample_points[indices_full]
    reconstructed_full = reconstruct_frobenius(sample_points_full, approximation_points, original_surface_values[indices_full, :], interpolant_degree, approx_basis) if len(sample_points_full) > 0 else np.zeros((len(approximation_points), surface_count))
    reconstructed_full = np.delete(reconstructed_full, -1, axis=1)

    if use_delaunay:
        triangulation_points = np.vstack([matched_cusp_points, hypercuboid_vertices(domains)])
        triangulation = Delaunay(triangulation_points)
        simplices = triangulation.simplices

        indices_delaunay = [[] for _ in range(len(simplices))]
        for i in indices_bottom:
            simplex_index = triangulation.find_simplex(sample_points[i])
            if simplex_index != -1:
                indices_delaunay[simplex_index].append(i)
        for index in indices_delaunay:
            assert len(index) > 0 # this is because the simplices triangulate the entire domain

        sample_points_delaunay = [sample_points[idx] for idx in indices_delaunay]
        reconstructions_delaunay = [reconstruct_frobenius(sample_points_delaunay[i], approximation_points, original_surface_values[indices_delaunay[i], 0:-1], interpolant_degree, approx_basis) for i in range(len(indices_delaunay))]

        weights_delaunay = [np.tile([simplex_approx_weight(delta_p, delta_q, triangulation_points, triangulation, x, j) for x in approximation_points], (surface_count - 1, 1)).T for j in range(len(simplices))]
        alpha = np.tile([matched_cusps_approx_weight(delta_p, delta_q, matched_cusp_points, x) for x in approximation_points], (surface_count - 1, 1))
        sum_weights = (1 - alpha) + sum(weights_delaunay)
        assert sum_weights.shape == reconstructed_full.shape

        plt_fn = lambda: plt.triplot(np.array(triangulation_points)[:, 0], np.array(triangulation_points)[:, 1], triangulation.simplices, color="red")
        return (reconstructed_full * (1 - alpha) + sum(reconstruction * weights for (reconstruction, weights) in zip(reconstructions_delaunay, weights_delaunay))) / sum_weights, plt_fn

    sample_points_bottom = sample_points[indices_bottom]
    reconstructed_bottom = reconstruct_frobenius(sample_points_bottom, approximation_points, original_surface_values[indices_bottom, 0:-1], interpolant_degree, approx_basis) if len(sample_points_bottom) > 0 else np.zeros((len(approximation_points), surface_count - 1))
    
    weights = np.tile([matched_cusps_approx_weight(delta_p, delta_q, matched_cusp_points, x) for x in approximation_points], (surface_count - 1, 1)).T
    assert reconstructed_full.shape == reconstructed_bottom.shape
    assert reconstructed_full.shape == weights.shape

    return reconstructed_bottom * weights + reconstructed_full * (1 - weights), lambda: ()

def reconstruct_direct(
    sample_points: FloatArray,
    approximation_points: FloatArray,
    original_surface_values: FloatArray,
    interpolant_degree: int,
    approx_basis: Basis
) -> FloatArray:
    assert sample_points.ndim == 2
    assert approximation_points.ndim == 2
    assert original_surface_values.ndim == 2
    assert sample_points.shape[0] == original_surface_values.shape[0]
    assert sample_points.shape[1] == approximation_points.shape[1]

    approxes = [points_approx(approx_basis, sample_points, surface, interpolant_degree) for surface in original_surface_values.T]
    reconstructed_surfaces = [[approx[0](x) for x in approximation_points] for approx in approxes]
    return np.array(reconstructed_surfaces)