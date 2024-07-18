

import numpy as np

from cusps.approx import points_approx
from cusps.esps import elem_sym_poly, reconstruct_roots


def estimate_matched_cusp_point(close_points): # given a set of sample points at which surfaces are close (potential matched cusp point candidates), estimate a representative for this set of points
    return min(close_points, key=lambda x: x[1])[0]

# TODO: write code for finding ubnmatched cusps using derivaives, adjust code for smoothing only there

def cluster_and_estimate(close_points: list[tuple], point_tolerance: float):
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

# TODO: add pyvista plot for bubble wrap function and one where the bubbles overlap