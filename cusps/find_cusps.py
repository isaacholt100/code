from typing import Any, TypedDict
import numpy as np

from bases import Basis
from cusps.approx import points_approx
from cusps.esps import elem_sym_poly, reconstruct_roots
from custom_types import FloatArray

class MatchCuspParams(TypedDict):
    value_tolerance: float
    point_tolerance: float

def estimate_matched_cusp_point(close_points: list[tuple]) -> FloatArray: # given a set of sample points at which surfaces are close (potential matched cusp point candidates), estimate a representative for this set of points
    return min(close_points, key=lambda x: x[1])[0]

def cluster_and_estimate(close_points: list[tuple[FloatArray, float]], point_tolerance: float) -> list[FloatArray]:
    assert all(c[0].ndim == 1 for c in close_points)

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

def find_matched_cusp_points(points: FloatArray, approx_surface_values: FloatArray, params: MatchCuspParams) -> list[FloatArray]:
    assert points.ndim == 2
    assert approx_surface_values.ndim == 2
    assert points.shape[0] == approx_surface_values.shape[0]

    close_points = []
    for point, roots in zip(points, approx_surface_values):
        dist = abs(roots[-2] - roots[-1])
        if dist < params["value_tolerance"]:
            close_points.append((point, dist))
    matched_cusp_points = cluster_and_estimate(close_points, params["point_tolerance"])
    return matched_cusp_points

def find_lower_matched_cusp_points(points: FloatArray, approx_surface_values: FloatArray, params: MatchCuspParams) -> list[FloatArray]:
    assert points.ndim == 2
    assert approx_surface_values.ndim == 2
    assert points.shape[0] == approx_surface_values.shape[0]

    close_points = []
    for point, roots in zip(points, approx_surface_values):
        dist = min(abs(roots[i] - roots[i + 1]) for i in range(len(roots) - 1))
        if dist < params["value_tolerance"]:
            close_points.append((point, dist))
    matched_cusp_points = cluster_and_estimate(close_points, params["point_tolerance"])
    return matched_cusp_points

def reconstruct_frobenius(
    sample_points: FloatArray,
    approximation_points: FloatArray,
    original_surface_values: FloatArray,
    interpolant_degree: int,
    approx_basis: Basis
) -> FloatArray:
    assert sample_points.ndim == 2
    assert approximation_points.ndim == 2
    assert sample_points.shape[1] == approximation_points.shape[1]
    assert sample_points.shape[0] == original_surface_values.shape[0]

    surface_count = original_surface_values.shape[1]
    esps = [elem_sym_poly(surface_count, r) for r in range(1, surface_count + 1)]
    esp_curves = [[esp(surface_value.tolist()) for surface_value in original_surface_values] for esp in esps]
    approxes = [points_approx(approx_basis, sample_points, np.array(esp_curve), interpolant_degree) for esp_curve in esp_curves]
    esp_values = [[approx[0](x) for approx in approxes] for x in approximation_points]
    
    reconstructed_surfaces = np.empty((len(approximation_points), surface_count))
    for i, values in enumerate(esp_values):
        roots = reconstruct_roots(values)
        # roots[-1] = math.nan
        reconstructed_surfaces[i] = roots

    return reconstructed_surfaces