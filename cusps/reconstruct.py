from typing import Any, TypedDict
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import Delaunay

from bases import Basis, BasisFunction, chebyshev_t
from cusps.approx import points_approx
from cusps.esps import elem_sym_poly, reconstruct_roots
from cusps.find_cusps import MatchCuspParams, find_lower_matched_cusp_points, find_matched_cusp_points, reconstruct_frobenius
from cusps.transition import matched_cusps_approx_weight, simplex_approx_weight
from custom_types import Curves, FloatArray
from plot_2d import plot_2d

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

class Indices(TypedDict):
    indices: list[int]
    weights: list[float]
    use_all_esps: bool
    direct: bool

def reconstruct_generalised(
    domains: list[tuple[float, float]],
    sample_points: FloatArray,
    approximation_points: FloatArray,
    original_surface_values: FloatArray,
    interpolant_degree: int,
    indices_list: list[Indices],
    approx_basis: Basis
) -> FloatArray:
    assert sample_points.ndim == 2
    assert approximation_points.ndim == 2
    assert original_surface_values.ndim == 2
    assert sample_points.shape[0] == original_surface_values.shape[0]
    assert sample_points.shape[1] == approximation_points.shape[1]
    assert sample_points.shape[1] == len(domains)
    assert all(len(indices["weights"]) == len(approximation_points) for indices in indices_list)

    assert interpolant_degree > 0

    dimension = len(domains)
    assert dimension > 0
    surface_count = original_surface_values.shape[1]

    approximations = [
        (reconstruct_direct(
            sample_points[indices["indices"]],
            approximation_points,
            original_surface_values[indices["indices"], :] if indices["use_all_esps"] else original_surface_values[indices["indices"], 0:-1],
            interpolant_degree,
            approx_basis
        ) if indices["direct"] else reconstruct_frobenius(
            sample_points[indices["indices"]],
            approximation_points,
            original_surface_values[indices["indices"], :] if indices["use_all_esps"] else original_surface_values[indices["indices"], 0:-1],
            interpolant_degree,
            approx_basis
        )) if len(indices) > 0 else np.zeros((len(approximation_points), surface_count))
        for indices in indices_list
    ]
    for i, indices in enumerate(indices_list):
        if indices["use_all_esps"]:
            approximations[i] = np.delete(approximations[i], -1, axis=1)

    weights_list: list[FloatArray] = [np.tile(indices["weights"], (surface_count - 1, 1)).T for indices in indices_list]

    return np.sum([approximation * weights for (approximation, weights) in zip(approximations, weights_list)], axis=0) / np.sum(weights_list, axis=0)

def reconstruct_ultra(
    domains: list[tuple[float, float]],
    sample_points: FloatArray,
    approximation_points: FloatArray,
    sample_values: FloatArray,
    interpolant_degree: int,
    delta_top: float,
    delta_lower: float,
    delta_away_from: float,
    approx_basis: Basis,
    match_top_cusps: MatchCuspParams | list[FloatArray],
    match_lower_cusps: MatchCuspParams | list[FloatArray]
) -> FloatArray:
    assert sample_points.ndim == 2
    assert approximation_points.ndim == 2
    assert sample_values.ndim == 2
    assert sample_points.shape[0] == sample_values.shape[0]
    assert sample_points.shape[1] == approximation_points.shape[1]
    assert sample_points.shape[1] == len(domains)

    assert interpolant_degree > 0
    assert delta_away_from >= 0
    assert delta_lower >= 0
    assert delta_top >= delta_away_from
    
    reconstructed_frob = reconstruct_frobenius(
        sample_points,
        approximation_points,
        sample_values,
        interpolant_degree,
        approx_basis
    )

    top_matched_cusp_points = find_matched_cusp_points(approximation_points, reconstructed_frob, match_top_cusps) if isinstance(match_top_cusps, dict) else match_top_cusps
    print("top cusps:", top_matched_cusp_points)
    lower_matched_cusp_points = find_lower_matched_cusp_points(approximation_points, reconstructed_frob, match_lower_cusps, exclude_top_surface=True) if isinstance(match_lower_cusps, dict) else match_lower_cusps
    # lower_matched_cusp_points = [np.array([0.0]), np.array([np.pi/6]), np.array([5*np.pi/8])]
    print("lower cusps:", lower_matched_cusp_points)

    top_cusps_indices_list: list[list[int]] = [[] for _ in top_matched_cusp_points]
    lower_cusps_indices_list: list[list[int]] = [[] for _ in lower_matched_cusp_points]
    away_from_top_cusps_indices: list[int] = []

    for i in range(len(sample_points)):
        for j, cusp in enumerate(top_matched_cusp_points):
            if np.linalg.norm(sample_points[i] - cusp) <= delta_top:
                top_cusps_indices_list[j].append(i)
        for j, cusp in enumerate(lower_matched_cusp_points):
            if np.linalg.norm(sample_points[i] - cusp) <= delta_lower:
                lower_cusps_indices_list[j].append(i)
        if all(np.linalg.norm(sample_points[i] - cusp) >= delta_away_from for cusp in top_matched_cusp_points):
            away_from_top_cusps_indices.append(i)

    top_cusps_weights = [[(1 - matched_cusps_approx_weight(delta_top - 0.01, delta_top, [cusp], x)) for x in approximation_points] for cusp in top_matched_cusp_points]
    lower_cusps_weights = [[(1 - matched_cusps_approx_weight(delta_lower - 0.01, delta_lower, [cusp], x)) for x in approximation_points] for cusp in lower_matched_cusp_points]
    away_from_top_cusps_weights = [matched_cusps_approx_weight(delta_top - 0.01, delta_top, top_matched_cusp_points, x) * matched_cusps_approx_weight(delta_lower - 0.01, delta_lower, lower_matched_cusp_points, x) for x in approximation_points]

    indices_list: list[Indices] = [{
        "indices": indices,
        "weights": weights,
        "use_all_esps": True,
        "direct": False,
    } for (indices, weights) in zip(top_cusps_indices_list, top_cusps_weights)]
    indices_list.extend({
        "direct": False,
        "use_all_esps": False,
        "indices": indices,
        "weights": weights
    } for (indices, weights) in zip(lower_cusps_indices_list, lower_cusps_weights))
    indices_list.append({
        "direct": False,
        "use_all_esps": False,
        "indices": away_from_top_cusps_indices,
        "weights": away_from_top_cusps_weights
    })

    return reconstruct_generalised(
        domains,
        sample_points,
        approximation_points,
        sample_values,
        interpolant_degree,
        indices_list,
        approx_basis
    )

def reconstruct_3_split_set(
    domains: list[tuple[float, float]],
    sample_points: FloatArray,
    approximation_points: FloatArray,
    sample_values: FloatArray,
    interpolant_degree: int,
    delta_inner: float,
    delta_outer: float,
    delta_lower_inner: float,
    delta_lower_outer: float,
    delta_away_from: float,
    approx_basis: Basis,
    match_top_cusps: MatchCuspParams | list[FloatArray],
    match_lower_cusps: MatchCuspParams | list[FloatArray]
) -> FloatArray:
    assert sample_points.ndim == 2
    assert approximation_points.ndim == 2
    assert sample_values.ndim == 2
    assert sample_points.shape[0] == sample_values.shape[0]
    assert sample_points.shape[1] == approximation_points.shape[1]
    assert sample_points.shape[1] == len(domains)

    assert interpolant_degree > 0
    assert delta_inner >= 0
    assert delta_outer >= delta_inner
    
    reconstructed_frob = reconstruct_frobenius(
        sample_points,
        approximation_points,
        sample_values,
        interpolant_degree,
        approx_basis
    )

    top_matched_cusp_points = find_matched_cusp_points(approximation_points, reconstructed_frob, match_top_cusps) if isinstance(match_top_cusps, dict) else match_top_cusps
    print("top cusps:", top_matched_cusp_points)
    lower_matched_cusp_points = find_lower_matched_cusp_points(approximation_points, reconstructed_frob, match_lower_cusps) if isinstance(match_lower_cusps, dict) else match_lower_cusps
    lower_matched_cusp_points = [np.array([0.0]), np.array([np.pi/6]), np.array([5*np.pi/8])]
    print("lower cusps:", lower_matched_cusp_points)

    indices_full = []
    indices_bottom = []
    indices_lower = []
    for i in range(len(sample_points)):
        if all(np.linalg.norm(sample_points[i] - p) >= delta_away_from for p in top_matched_cusp_points):
            indices_bottom.append(i)
        if any(np.linalg.norm(sample_points[i] - p) <= delta_outer for p in top_matched_cusp_points):
            indices_full.append(i)
        if any(np.linalg.norm(sample_points[i] - p) <= delta_outer for p in lower_matched_cusp_points):
            indices_lower.append(i)

    top_cusps_weights = [(1 - matched_cusps_approx_weight(delta_inner, delta_outer, top_matched_cusp_points, x)) for x in approximation_points]
    lower_cusps_weights = [(1 - matched_cusps_approx_weight(delta_lower_inner, delta_lower_outer, lower_matched_cusp_points, x)) for x in approximation_points]
    away_from_top_cusps_weights = [matched_cusps_approx_weight(delta_inner, delta_inner, top_matched_cusp_points, x) * matched_cusps_approx_weight(delta_lower_inner, delta_lower_inner, lower_matched_cusp_points, x) for x in approximation_points]

    indices_list: list[Indices] = [{
        "indices": indices_full,
        "weights": top_cusps_weights,
        "use_all_esps": True,
        "direct": False,
    }, {
        "indices": indices_lower,
        "weights": lower_cusps_weights,
        "use_all_esps": False,
        "direct": False,
    }, {
        "indices": indices_bottom,
        "weights": away_from_top_cusps_weights,
        "use_all_esps": False,
        "direct": False
    }]

    return reconstruct_generalised(
        domains,
        sample_points,
        approximation_points,
        sample_values,
        interpolant_degree,
        indices_list,
        approx_basis
    )

def reconstruct_enhanced(
    domains: list[tuple[float, float]],
    sample_points: FloatArray,
    approximation_points: FloatArray,
    original_surface_values: FloatArray,
    interpolant_degree: int,
    delta_inner: float,
    delta_outer: float,
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
    assert delta_inner >= 0
    assert delta_outer >= delta_inner
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
    lower_matched_cusp_points = find_lower_matched_cusp_points(approximation_points, reconstructed_frob, {"value_tolerance": 0.1, "point_tolerance": 0.11})

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
    indices_lower = []
    for i in range(len(sample_points)):
        if all(np.linalg.norm(sample_points[i] - p) >= delta_inner for p in matched_cusp_points):
            indices_bottom.append(i)
        if any(np.linalg.norm(sample_points[i] - p) <= delta_outer for p in matched_cusp_points):
            indices_full.append(i)
        if any(np.linalg.norm(sample_points[i] - p) <= delta_outer for p in lower_matched_cusp_points):
            indices_lower.append(i)

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

        alpha = [matched_cusps_approx_weight(delta_inner, delta_outer, matched_cusp_points, x) for x in approximation_points]
        indices_list: list[Indices] = [{
            "indices": indices_delaunay[i],
            "use_all_esps": False,
            "weights": [simplex_approx_weight(delta_inner, delta_outer, triangulation_points, triangulation, x, i) for x in approximation_points],
            "direct": False,
        } for i in range(len(simplices))]
        indices_list.append({
            "indices": indices_full,
            "use_all_esps": True,
            "weights": [1 - weight for weight in alpha],
            "direct": False,
        })

        plt_fn = lambda: plt.triplot(np.array(triangulation_points)[:, 0], np.array(triangulation_points)[:, 1], triangulation.simplices, color="red")
        return reconstruct_generalised(
            domains,
            sample_points,
            approximation_points,
            original_surface_values,
            interpolant_degree,
            indices_list,
            approx_basis
        ), plt_fn

    weights = [matched_cusps_approx_weight(delta_inner, delta_outer, matched_cusp_points, x) for x in approximation_points]
    # plt.plot(approximation_points, weights)
    # plt.show()
    # plot_2d(approximation_points, np.array([weights]).T)
    # plt.plot(approximation_points, weights)
    # plt.show()
    # plt.scatter(*zip(*sample_points[indices_bottom]))
    # plt.scatter(*zip(*sample_points[indices_full]))
    # plt.show()
    indices_list: list[Indices] = [
        {
            "indices": indices_full,
            "use_all_esps": True,
            "weights": [1 - weight for weight in weights],
            "direct": False,
        },
        {
            "indices": indices_bottom,
            "use_all_esps": False,
            "weights": weights,
            "direct": False,
        },
        # {
        #     "indices": indices_lower,
        #     "use_all_esps": False,
        #     "weights": [1 - matched_cusps_approx_weight(delta_inner, delta_outer, lower_matched_cusp_points, x) for x in approximation_points],
        #     "direct": False
        # }
    ]

    return reconstruct_generalised(
        domains,
        sample_points,
        approximation_points,
        original_surface_values,
        interpolant_degree,
        indices_list,
        approx_basis
    ), lambda: ()

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

        weights_delaunay = [np.tile([simplex_approx_weight(delta_inner, delta_outer, triangulation_points, triangulation, x, j) for x in approximation_points], (surface_count - 1, 1)).T for j in range(len(simplices))]
        alpha = np.tile([matched_cusps_approx_weight(delta_inner, delta_outer, matched_cusp_points, x) for x in approximation_points], (surface_count - 1, 1)).T
        sum_weights = (1 - alpha) + sum(weights_delaunay)
        assert sum_weights.shape == reconstructed_full.shape

        plt_fn = lambda: plt.triplot(np.array(triangulation_points)[:, 0], np.array(triangulation_points)[:, 1], triangulation.simplices, color="red")
        return (reconstructed_full * (1 - alpha) + sum(reconstruction * weights for (reconstruction, weights) in zip(reconstructions_delaunay, weights_delaunay))) / sum_weights, plt_fn

    sample_points_bottom = sample_points[indices_bottom]
    reconstructed_bottom = reconstruct_frobenius(sample_points_bottom, approximation_points, original_surface_values[indices_bottom, 0:-1], interpolant_degree, approx_basis) if len(sample_points_bottom) > 0 else np.zeros((len(approximation_points), surface_count - 1))

    plt.scatter([x[0] for x in approximation_points[1:-1]], deriv(reconstructed_bottom[:, 1], 2.1/(4000 - 1)), s=0.2)
    # plt.plot(plot_points, reconstructed_surfaces[1, :])
    plt.scatter([x[0] for x in approximation_points[1:-1]], deriv(reconstructed_bottom[:, 0], 2.1/(4000 - 1)), s=0.2)
    plt.show()

    plt.scatter([x[0] for x in approximation_points[1:-1]], deriv(reconstructed_full[:, 1], 2.1/(4000 - 1)), s=0.2)
    # plt.plot(plot_points, reconstructed_surfaces[1, :])
    plt.scatter([x[0] for x in approximation_points[1:-1]], deriv(reconstructed_full[:, 0], 2.1/(4000 - 1)), s=0.2)
    plt.show()
    
    weights = np.tile([matched_cusps_approx_weight(delta_inner, delta_outer, matched_cusp_points, x) for x in approximation_points], (surface_count - 1, 1)).T

    plt.scatter([x[0] for x in approximation_points[1:-1]], deriv(weights[:, 1], 2.1/(4000 - 1)), s=0.2)
    # plt.plot(plot_points, reconstructed_surfaces[1, :])
    plt.scatter([x[0] for x in approximation_points[1:-1]], deriv(weights[:, 0], 2.1/(4000 - 1)), s=0.2)
    plt.show()

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
    reconstructed_surfaces = [[approx[0](x) for approx in approxes] for x in approximation_points]
    return np.array(reconstructed_surfaces)