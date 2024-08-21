from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import itertools
import math
import random
from typing import Iterable
import seaborn as sns
from scipy import stats

from matplotlib import pyplot as plt
import numpy as np
from bases import Basis
from cusps.errors import gap_weighted_error, gap_weighted_error_pointwise, sum_max_abs_error
from cusps.find_cusps import MatchCuspParams, find_lower_matched_cusp_points, find_matched_cusp_points, reconstruct_frobenius
from cusps.reconstruct import get_surface_values, reconstruct_enhanced
from custom_types import Curves, FloatArray
from plot_2d import plot_2d
from polyapprox.analyses import plotting_grid_points

def _angles_to_sphere_coords(angles: list[float]) -> list[float]:
    dim = len(angles)
    assert dim >= 1
    if dim == 1:
        return [math.cos(angles[0]), math.sin(angles[0])]
    prev_coords = _angles_to_sphere_coords(angles[0:-1])
    l = prev_coords[-1]
    prev_coords[-1] *= math.cos(angles[-1])
    prev_coords.append(math.sin(angles[-1]) * l)
    return prev_coords

def _random_point_in_ball(centre: FloatArray, radius: float) -> FloatArray: # generate uniformly random points in a ball
    assert centre.ndim == 1

    if len(centre) == 1:
        return np.array([radius * (2*random.random() - 1) + centre[0]])
    assert len(centre) > 1
    point_radius = random.random() * radius
    angle_first = random.random() * 2 * math.pi
    angles = [random.random() * math.pi for _ in range(len(centre) - 2)]
    angles.insert(0, angle_first)
    # angle_last = random.random() * 2 * math.pi
    return np.array([c + point_radius * coord for (c, coord) in zip(centre, _angles_to_sphere_coords(angles))])

def _request_random_points_at_cusp(cusp: FloatArray, delta_outer: float, count: int, dist) -> FloatArray:
    assert cusp.ndim == 1
    if dist == "eq-space":
        if len(cusp) == 1:
            return np.transpose([np.linspace(cusp[0] - delta_outer, cusp[0] + delta_outer, count)])
        return np.array(list(filter(lambda p: np.linalg.norm(p - cusp) <= delta_outer, plotting_grid_points([(cusp[0] - delta_outer, cusp[0] + delta_outer), (cusp[1] - delta_outer, cusp[1] + delta_outer)], math.ceil(count * 4/np.pi)))))
    if dist == "normal":
        # ~99.7% of values drawn from normal distribution are within 3 standard deviations of the mean
        if len(cusp) == 1:
            return np.transpose([stats.norm.rvs(cusp[0], delta_outer/3, size=count)])
        return np.transpose([stats.norm.rvs(cusp[0], delta_outer/3, size=count), stats.norm.rvs(cusp[1], delta_outer/3, size=count)])
    return np.array([_random_point_in_ball(cusp, delta_outer) for _ in range(count)])

def request_random_points(matched_cusps: list[FloatArray], delta_outer: float, count: int, dist) -> FloatArray:
    assert all(cusp.ndim == 1 for cusp in matched_cusps)

    single_count = count // len(matched_cusps)
    last_count = single_count + (count % len(matched_cusps))
    assert single_count * (len(matched_cusps) - 1) + last_count == count

    points = _request_random_points_at_cusp(matched_cusps[0], delta_outer, last_count, dist)
    for cusp in matched_cusps[1:]:
        new_points = _request_random_points_at_cusp(cusp, delta_outer, single_count, dist)
        points = np.vstack([points, new_points])
    return points

def is_in_approx_domain(point: FloatArray, domains: list[tuple[float, float]]) -> bool:
    assert point.ndim == 1
    assert len(point) == len(domains)
    
    for i in range(len(point)):
        a, b = domains[i]
        if a > point[i] or point[i] > b:
            return False
    return True

def request_random_sample_points_and_values(
    curves: Curves,
    matched_cusps: list[FloatArray],
    delta_outer: float,
    count: int,
    domains: list[tuple[float, float]],
    dist
) -> tuple[FloatArray, FloatArray]:
    points = request_random_points(matched_cusps, delta_outer, count, dist)
    print(points)
    points = np.array(list(filter(lambda point: is_in_approx_domain(point, domains), 
    points)))
    return points, get_surface_values(curves, points)

# def request_equispaced_points

def reconstruct_active(
    err_diff_tol: float,
    new_points_count: int,
    sample_radius: float,
    dist,
    max_iters: int,
    domains: list[tuple[float, float]],
    sample_points: FloatArray,
    approximation_points: FloatArray,
    curves: Curves,
    interpolant_degree: int,
    delta_inner: float,
    delta_outer: float,
    removal_epsilon: float,
    approx_basis: Basis,
    match_cusps: MatchCuspParams,
    use_delaunay = False
) -> FloatArray:
    assert sample_points.ndim == 2
    assert approximation_points.ndim == 2
    assert sample_points.shape[1] == approximation_points.shape[1]
    assert sample_points.shape[1] == len(domains)

    sample_surface_values = get_surface_values(curves, sample_points)
    approximation_surface_values = get_surface_values(curves, approximation_points)
    reconstructed_frob = reconstruct_frobenius(
        sample_points,
        approximation_points,
        sample_surface_values,
        interpolant_degree,
        approx_basis
    )
    matched_cusp_points = find_matched_cusp_points(approximation_points, reconstructed_frob, match_cusps)
    # plt.scatter(*zip(*matched_cusp_points))
    plt.show()
    prev_reconstruction, _ = reconstruct_enhanced(
        domains,
        sample_points,
        approximation_points,
        sample_surface_values,
        interpolant_degree,
        delta_inner,
        delta_outer,
        removal_epsilon,
        approx_basis,
        matched_cusp_points,
        use_delaunay
    )
    # plt.plot(approximation_points, np.log10(gap_weighted_error_pointwise(approximation_surface_values, prev_reconstruction)), label=f"0 extra sample points")
    prev_error = gap_weighted_error(approximation_surface_values, prev_reconstruction)
    # plot_2d(approximation_points, prev_reconstruction)
    ## TODO: find new cusp candidates in each iteration, use the previous reconstruction to do this
    ## TODO: warn if endpoint * weighting function is too large (should be very close to 0)
    for i in range(max_iters):
        print("iters:", i)
        lower_matched_cusp_points = find_lower_matched_cusp_points(approximation_points, prev_reconstruction, match_cusps)
        print(prev_reconstruction.shape)
        print(len(lower_matched_cusp_points))
        # print("lower matched cusps:", lower_matched_cusp_points)
        extra_sample_points, extra_sample_values = request_random_sample_points_and_values(curves, lower_matched_cusp_points, sample_radius, new_points_count, domains, dist=dist)


        # plt.scatter(*zip(*matched_cusp_points))
        # plt.scatter(*zip(*extra_sample_points))
        # plt.show()
        sample_points = np.vstack([sample_points, extra_sample_points])
        # plt.scatter(sample_points, [1] * len(sample_points))
        sample_surface_values = np.vstack([sample_surface_values, extra_sample_values])
        # print(len(sample_points))
        new_reconstruction, _ = reconstruct_enhanced(
            domains,
            sample_points,
            approximation_points,
            sample_surface_values,
            interpolant_degree,
            delta_inner,
            delta_outer,
            removal_epsilon,
            approx_basis,
            matched_cusp_points,
            use_delaunay
        )
        new_error = gap_weighted_error(approximation_surface_values, new_reconstruction)
        print("new:", new_error, ", old:", prev_error)
        if abs(new_error - prev_error) < err_diff_tol:
            return new_reconstruction
        prev_reconstruction = new_reconstruction
        prev_error = new_error
    return prev_reconstruction

# play around with different distirubtions around cusp
# get overview on ML literature on data aug for rotations

def empirical_distribution_estimate(
    max_iters_list: list[int],
    sample_count: int,
    sample_retain_count: int,
    sample_radius: float,
    domains: list[tuple[float, float]],
    sample_points: FloatArray,
    approximation_points: FloatArray,
    curves: Curves,
    interpolant_degree: int,
    delta_inner: float,
    delta_outer: float,
    removal_epsilon: float,
    approx_basis: Basis,
    match_cusps: MatchCuspParams,
    use_delaunay = False
):
    assert sample_points.ndim == 2
    assert approximation_points.ndim == 2
    assert sample_points.shape[1] == approximation_points.shape[1]
    assert sample_points.shape[1] == len(domains)

    included_samples = []
    sample_surface_values = get_surface_values(curves, sample_points)
    approximation_surface_values = get_surface_values(curves, approximation_points)
    reconstructed_frob = reconstruct_frobenius(
        sample_points,
        approximation_points,
        sample_surface_values,
        interpolant_degree,
        approx_basis
    )
    matched_cusp_points = find_matched_cusp_points(approximation_points, reconstructed_frob, match_cusps)
    prev_reconstruction, _ = reconstruct_enhanced(
        domains,
        sample_points,
        approximation_points,
        sample_surface_values,
        interpolant_degree,
        delta_inner,
        delta_outer,
        removal_epsilon,
        approx_basis,
        matched_cusp_points,
        use_delaunay
    )
    ## TODO: warn if endpoint * weighting function is too large (should be very close to 0)
    curves_sequence = []
    gap_weighted_errors_sequence = []
    max_abs_errors_sequence = []
    lower_matched_cusp_points = find_lower_matched_cusp_points(approximation_points, prev_reconstruction, match_cusps)#[np.array([0, 0])] if len(domains) == 2 else [np.array([5*np.pi/8])]
    test_points, test_values = request_random_sample_points_and_values(curves, lower_matched_cusp_points, sample_radius, 4*sample_count, domains, dist="eq-space")
    print("matched_cusp_points:", matched_cusp_points)
    for i in range(max(max_iters_list)):
        # lower_matched_cusp_points = find_lower_matched_cusp_points(approximation_points, prev_reconstruction, match_cusps)
        print("iters:", i)
        # lower_matched_cusp_points = find_lower_matched_cusp_points(approximation_points, prev_reconstruction, match_cusps)
        extra_sample_points, extra_sample_values = request_random_sample_points_and_values(curves, lower_matched_cusp_points, sample_radius, sample_count, domains, dist="eq-space")
        temp_sample_points = np.vstack([sample_points, extra_sample_points, lower_matched_cusp_points])
        temp_sample_values = np.vstack([sample_surface_values, extra_sample_values, get_surface_values(curves, np.array(lower_matched_cusp_points))])
        new_reconstruction, _ = reconstruct_enhanced(
            domains,
            temp_sample_points,
            test_points,
            temp_sample_values,
            interpolant_degree,
            delta_inner,
            delta_outer,
            removal_epsilon,
            approx_basis,
            matched_cusp_points,
            use_delaunay
        )
        prev_reconstruction = new_reconstruction
        errors = []
        assert len(test_values) == len(test_points) and len(test_points) == len(new_reconstruction)
        for sample_point, sample_value, estimated_value in zip(test_points, test_values, new_reconstruction):
            error = max(abs(estimated_value[j] - sample_value[j]) for j in range(len(estimated_value)))
            errors.append((error, sample_point, sample_value))
        errors.sort(key=lambda a: a[0], reverse=True)
        retain_count = sample_retain_count#round(((max_iters - i)/max_iters) * (sample_retain_count - 1)) + 1
        for (error, sample_point, sample_value) in itertools.islice(errors, retain_count):
            print("error:", error)
            print(sample_point - lower_matched_cusp_points[0])
            included_samples.append(sample_point - lower_matched_cusp_points[0])
            sample_points = np.vstack([sample_points, sample_point])
            sample_surface_values = np.vstack([sample_surface_values, sample_value])
            # continue

            idx = np.where((test_points == sample_point).all(axis=1))[0][0]
            assert np.array_equal(test_points[idx], sample_point)
            test_points = np.delete(test_points, idx, axis=0)
            test_values = np.delete(test_values, idx, axis=0)
            min_dist = np.inf
            for test_point in test_points:
                if np.array_equal(test_point, sample_point):
                    continue
                dist = np.linalg.norm(test_point - sample_point)
                if dist < min_dist:
                    min_dist = dist
            for j in range(len(sample_point)):
                new_point = sample_point.copy()
                new_point[j] += min_dist/2
                test_points = np.vstack([test_points, new_point])
                test_values = np.vstack([test_values, curves(*new_point)])
                new_point[j] -= min_dist
                test_points = np.vstack([test_points, new_point])
                test_values = np.vstack([test_values, curves(*new_point)])
        if i + 1 in max_iters_list:
            new_reconstruction, _ = reconstruct_enhanced(
                domains,
                sample_points,
                approximation_points,
                sample_surface_values,
                interpolant_degree,
                delta_inner,
                delta_outer,
                removal_epsilon,
                approx_basis,
                matched_cusp_points,
                use_delaunay
            )
            gap_weighted_errors_sequence.append(gap_weighted_error(approximation_surface_values, new_reconstruction))
            max_abs_errors_sequence.append(sum_max_abs_error(approximation_surface_values, new_reconstruction))
            curves_sequence.append(deepcopy(included_samples))

    plt.plot(max_iters_list, np.log10(gap_weighted_errors_sequence))
    plt.xlabel("number of iterations")
    plt.ylabel("log 10 of gap weighted error")
    plt.title("convergence plot")
    plt.show()
    plt.plot(max_iters_list, np.log10(max_abs_errors_sequence))
    plt.xlabel("number of iterations")
    plt.ylabel("log 10 of max abs error")
    plt.title("convergence plot")
    plt.show()
    plt.figure(figsize=(12, 6))
    data_sequence = [[np.linalg.norm(x) for x in samples] for samples in curves_sequence]
    print(len(data_sequence), len(max_iters_list))
    assert len(data_sequence) == len(max_iters_list)
    minmax_sequence = []
    for (max_iters, data) in zip(max_iters_list, data_sequence):
        sns_kde = sns.kdeplot(data, label=f"kde ({max_iters} iters)", common_norm=False)
        x_vals = sns_kde.get_lines()[-1].get_xdata()
        minmax_sequence.append((np.min(x_vals), np.max(x_vals)))
        y_vals = sns_kde.get_lines()[-1].get_ydata()
        plt.scatter(x_vals, y_vals, color=sns_kde.get_lines()[-1].get_color(), s=10)
    plt.legend()
    plt.title("linear-linear")
    plt.show()
    from scipy.stats import gaussian_kde
    from scipy.integrate import simpson
    for (max_iters, minmax, data) in zip(max_iters_list, minmax_sequence, data_sequence):
        kde = gaussian_kde(data)
        x_vals = np.linspace(0, minmax[1], 1000)
        x_vals = np.hstack([x_vals, np.linspace(0, 0.0002, 1000)])
        x_vals = np.sort(x_vals)
        print(x_vals.shape)
        plt.plot(x_vals, kde(x_vals), label=f"kde ({max_iters} iters)")
        print(kde.integrate_box(minmax[0], minmax[1]))
    plt.legend()
    plt.title("linear-linear (fine grained centre)")
    plt.show()
    # for (max_iters, data) in zip(max_iters_list, data_sequence):
    #     sns.kdeplot(data, label=f"log kde ({max_iters} iters)", log_scale=(False, True))
    # plt.legend()
    # plt.title("normal log")
    # plt.show()
    for (max_iters, minmax, data) in zip(max_iters_list, minmax_sequence, data_sequence):
        kde = gaussian_kde(data)
        x_vals = np.linspace(minmax[0], minmax[1], 1000)
        x_vals = np.hstack([x_vals, np.linspace(0, 0.0002, 1000)])
        x_vals = np.sort(x_vals)
        plt.plot(np.log10(x_vals), np.log10(kde(x_vals)), label=f"kde ({max_iters} iters)")
    plt.legend()
    plt.title("log-log")
    plt.show()