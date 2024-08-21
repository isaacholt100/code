import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

from bases import cheb_basis
from cusps import surfaces
from cusps.active import empirical_distribution_estimate, reconstruct_active, request_random_points
from cusps.errors import gap_weighted_error, gap_weighted_error_pointwise, sum_abs_error_pointwise, sum_max_abs_error
from cusps.esps import elem_sym_poly, reconstruct_roots, reconstruct_roots_chebyshev
from cusps.find_cusps import reconstruct_frobenius
from cusps.reconstruct import cheb_basis_shifted, deriv, find_matched_cusp_points, get_surface_values, reconstruct_enhanced, reconstruct_ultra
from cusps.surfaces import curves_1, sample_values_with_noise, surfaces_with_noise
from custom_types import Curves, FloatArray
from polyapprox.analyses import plotting_grid_points

def rand_unif_samples(a: float, b: float, sample_size: int) -> FloatArray:
    assert sample_size > 0
    return np.sort(np.random.uniform(a, b, sample_size))

samples = np.transpose([rand_unif_samples(0, 2, 4000)])#np.transpose([np.linspace(0, 2, 67)])

def plot_reconstruction(x_values, original_surfaces, reconstructed_surfaces, name: str):
    plt.figure(0)
    for i in range(original_surfaces.shape[1] - 1):
        plt.plot(x_values, original_surfaces[:, i])
    plt.title("Original multi-surface data")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

    print(name, reconstructed_surfaces.shape)

    plt.figure(1)
    for i in range(reconstructed_surfaces.shape[1]):
        plt.plot(x_values, reconstructed_surfaces[:, i])
    plt.title(f"ESP-reconstructed multi-surface ({name})")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

    errors = [np.abs(reconstructed_surfaces[:, i] - original_surfaces[:, i]) for i in range(reconstructed_surfaces.shape[1])]
    for error_points in errors:
        plt.plot(x_values, np.log10(error_points))
    plt.ylabel("log10 of abs error")
    plt.title(f"Error plot ({name})")
    plt.show()

    errors = gap_weighted_error_pointwise(original_surfaces, reconstructed_surfaces)
    plt.plot(x_values, np.log10(errors))
    plt.ylabel("log10 of pointwise gap weighted error")
    plt.title(f"Gap-weighted error plot ({name})")
    plt.show()

def plot_params():
    values = np.linspace(0.0, 0.5, 10)
    max_abs_errors, gap_weighted_errors = [], []
    for value in values:
        original_surfaces = get_surface_values(curves_1, samples)
        reconstructed_surfaces, _ = reconstruct_enhanced([(0, 2)], samples, samples, original_surface_values=original_surfaces, interpolant_degree=20, delta_inner=0.15, delta_outer=0.268+value, removal_epsilon=0.0, approx_basis=cheb_basis_shifted(0, 2), match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.1})
        max_abs_errors.append(sum_max_abs_error(original_surfaces, reconstructed_surfaces))
        gap_weighted_errors.append(gap_weighted_error(original_surfaces, reconstructed_surfaces))

    plt.plot(values, np.log10(max_abs_errors))
    plt.ylabel("max abs error")
    plt.xlabel("delta_inner")
    plt.show()
    plt.plot(values, np.log10(gap_weighted_errors))
    plt.ylabel("gap weighted error")
    plt.xlabel("delta_inner")
    plt.show()
# plot_params()

plot_points = np.transpose([np.linspace(0, 2, 4000)])

# TODO: interpolation and quadrature rule for spherical harmonics

sample_surface_values = sample_values_with_noise(0.00000, get_surface_values(curves_1, samples))
plot_surface_values = get_surface_values(curves_1, plot_points)

## TODO: think about method where we only use two ESPs to reconstruct around each matched cusp (top and lower level)

def plot_error_against_noise(sample_points, plot_points, curves, noise_levels):
    sample_values_no_noise = get_surface_values(curves, sample_points)
    plot_values = get_surface_values(curves, plot_points)
    gap_weighted_errors_enhanced = []
    max_abs_errors_enhanced = []
    gap_weighted_errors_frobenius = []
    max_abs_errors_frobenius = []
    for noise_level in noise_levels:
        noisy_samples = sample_values_with_noise(noise_level, sample_values_no_noise)
        reconstruction_frobenius = reconstruct_frobenius(
            sample_points=sample_points,
            approximation_points=plot_points,
            original_surface_values=noisy_samples,
            interpolant_degree=20,
            approx_basis=cheb_basis_shifted(0, 2)
        )
        reconstruction_enhanced, _ = reconstruct_enhanced(
            domains=[(0, 2)],
            sample_points=sample_points,
            approximation_points=plot_points,
            original_surface_values=noisy_samples,
            interpolant_degree=20,
            removal_epsilon=0.0,
            delta_inner=0.23,
            delta_outer=0.27,
            approx_basis=cheb_basis_shifted(0, 2),
            match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.11},
        )
        gap_weighted_errors_enhanced.append(gap_weighted_error(plot_values, reconstruction_enhanced))
        max_abs_errors_enhanced.append(sum_max_abs_error(plot_values, reconstruction_enhanced))
        
        gap_weighted_errors_frobenius.append(gap_weighted_error(plot_values, reconstruction_frobenius))
        max_abs_errors_frobenius.append(sum_max_abs_error(plot_values, reconstruction_frobenius))
    plt.plot(noise_levels, np.log10(gap_weighted_errors_enhanced), label="Split-set method")
    plt.plot(noise_levels, np.log10(gap_weighted_errors_frobenius), label="Frobenius method")
    plt.xlabel("noise level (standard deviation)")
    plt.ylabel("log10 of error")
    plt.title("Gap weighted error")
    plt.legend()
    plt.show()

    plt.plot(noise_levels, np.log10(max_abs_errors_enhanced), label="Split-set method")
    plt.plot(noise_levels, np.log10(max_abs_errors_frobenius), label="Frobenius method")
    plt.xlabel("noise level (standard deviation)")
    plt.ylabel("log10 of error")
    plt.title("Max abs error")
    plt.legend()
    plt.show()

plot_error_against_noise(samples, plot_points, curves_1, np.linspace(0, 0.01, 100))

exit(0)

# TODO: TRY various different distributions e.g. (spikey) normal, triangular, 1/r dist
# make it log plot on y axis
# TODO: try the two different methods that Christoph mentioned (on the sheet of paper)
# TODO: plot errors in the ESPs rather than the surfaces
# for max_iters in [*range(2, 10), 10, 20, 50, 100]:
# empirical_distribution_estimate([100], 1000, 10, 0.2, [(0, 2.4)], samples, plot_points, curves_1, interpolant_degree=20, delta_inner=0.20, delta_outer=0.27, removal_epsilon=0.0, approx_basis=cheb_basis_shifted(0, 2.4), match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.11})


# empirical_distribution_estimate(list(range(1, 102, 2)), 1000, 10, 0.2, [(0, 2.4)], samples, plot_points, curves_1, interpolant_degree=20, delta_inner=0.20, delta_outer=0.27, removal_epsilon=0.0, approx_basis=cheb_basis_shifted(0, 2.4), match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.11})

# empirical_distribution_estimate([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 1000, 10, 0.1, [(-1, 1), (-1, 1)], samples_2d, plot_points_2d, curves_sinh(1, 1), interpolant_degree=20, delta_inner=0.20, delta_outer=0.27, removal_epsilon=0.0, approx_basis=cheb_basis, match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.11})
# empirical_distribution_estimate([10, 20, 30, 40, 50], 1000, 10, 0.1, [(0, 2.4)], samples, plot_points, curves_1, interpolant_degree=20, delta_inner=0.20, delta_outer=0.27, removal_epsilon=0.0, approx_basis=cheb_basis, match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.11})

# exit(0)

## TODO: try directly reconstructing the surfaces in the Delaunay method rather than using ESPs, compare accuracy compared to using ESPs
## TODO: also have a think about better ways of triangulating. look at the one you wrote in your notepad. could be quicker than Delaunay

def plot_convergence(counts):
    gap_weighted_errors = []
    max_abs_errors = []
    for new_points_count in counts:
        # print(new_points_count)
        reconstruction = reconstruct_active(
            err_diff_tol=0,
            new_points_count=new_points_count,
            sample_radius=0.01,
            dist="normal",
            max_iters=1,
            domains=[(0, 2)],
            sample_points=samples,
            approximation_points=plot_points,
            curves=curves_1,
            interpolant_degree=20,
            delta_inner=0.268,
            delta_outer=0.270,
            removal_epsilon=0.0,
            approx_basis=cheb_basis_shifted(0, 2),
            match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.11}
        )
        gap_weighted_errors.append(gap_weighted_error(plot_surface_values, reconstruction))
        max_abs_errors.append(sum_max_abs_error(plot_surface_values, reconstruction))
    plt.plot(list(counts), np.log10(gap_weighted_errors))
    plt.xlabel("number of new sample points")
    plt.ylabel("log10 of gap weighted error")
    plt.show()


    plt.plot(list(counts), np.log10(max_abs_errors))
    plt.xlabel("number of new sample points")
    plt.ylabel("log10 of max abs error")
    plt.show()

# plot_convergence(range(1000, 10000, 500))

# exit(0)

def plot_sample_radius(radii):
    gap_weighted_errors = []
    max_abs_errors = []
    for radius in radii:
        reconstruction = reconstruct_active(
            err_diff_tol=0,
            new_points_count=1000,
            sample_radius=radius,
            dist="normal",
            max_iters=1,
            domains=[(0, 2.4)],
            sample_points=samples,
            approximation_points=plot_points,
            curves=curves_1,
            interpolant_degree=20,
            delta_inner=0.268,
            delta_outer=0.270,
            removal_epsilon=0.0,
            approx_basis=cheb_basis_shifted(0, 2.4),
            match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.11}
        )
        gap_weighted_errors.append(gap_weighted_error(plot_surface_values, reconstruction))
        max_abs_errors.append(sum_max_abs_error(plot_surface_values, reconstruction))
    plt.plot(list(radii), np.log10(gap_weighted_errors))
    plt.xlabel("sample radius")
    plt.ylabel("log10 of gap weighted error")
    plt.show()


    plt.plot(list(radii), np.log10(max_abs_errors))
    plt.xlabel("sample radius")
    plt.ylabel("log10 of max abs error")
    plt.show()

# plot_sample_radius(np.linspace(0.001, 1, 20))

# exit(0)
# for new_points_count in list(range(10, 11, 1000)):
#     reconstruction = reconstruct_active(
#         err_diff_tol=0,
#         new_points_count=new_points_count,
#         sample_radius=0.01,
#         dist="normal",
#         max_iters=1,
#         domains=[(0, 2)],
#         sample_points=samples,
#         approximation_points=plot_points,
#         curves=curves_1,
#         interpolant_degree=25,
#         delta_inner=0.268,
#         delta_outer=0.270,
#         removal_epsilon=0.0,
#         approx_basis=cheb_basis_shifted(0, 2),
#         match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.11}
#     )
#     plt.plot(plot_points, np.log10(gap_weighted_error_pointwise(plot_surface_values, reconstruction)), label=f"{new_points_count} extra sample points")

# plt.xlabel("x value")
# plt.ylabel("log10 of pointwise gap weighted error")
# plt.ylim(-16, 0)
# plt.legend()
# # plt.title("10000 extra sample points")
# plt.show()

noisy_surface_values = sample_values_with_noise(0.005, sample_surface_values)

# exit(0)

# reconstruction_frob = reconstruct_frobenius(
#     sample_points=samples,
#     approximation_points=plot_points,
#     original_surface_values=noisy_surface_values,
#     interpolant_degree=20,
#     approx_basis=cheb_basis_shifted(0, 2)
# )
# reconstruction_frob = np.delete(reconstruction_frob, 2, axis=1)
# plot_reconstruction(plot_points, plot_surface_values, reconstruction_frob, name="Original method")
# print("sum max abs error:", sum_max_abs_error(plot_surface_values, reconstruction_frob))
# print("gap weighted error:", gap_weighted_error(plot_surface_values, reconstruction_frob))

reconstruction_enhanced, _ = reconstruct_enhanced(
    domains=[(0, 2)],
    sample_points=samples,
    approximation_points=plot_points,
    original_surface_values=noisy_surface_values,
    interpolant_degree=20,
    removal_epsilon=0.0,
    delta_inner=0.268,
    delta_outer=0.27,
    approx_basis=cheb_basis_shifted(0, 2),
    match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.11},
)
plot_reconstruction(plot_points, plot_surface_values, reconstruction_enhanced, name="Split set")
print("sum max abs error:", sum_max_abs_error(plot_surface_values, reconstruction_enhanced))
print("gap weighted error:", gap_weighted_error(plot_surface_values, reconstruction_enhanced))

exit(0)

# reconstruction_ultra = reconstruct_ultra(
#     domains=[(0, 2)],
#     sample_points=samples,
#     approximation_points=plot_points,
#     sample_values=sample_surface_values,
#     interpolant_degree=20,
#     delta_top=0.2,
#     delta_lower=0.1,
#     delta_away_from=0.19,
#     approx_basis=cheb_basis_shifted(0, 2),
#     match_top_cusps={"value_tolerance": 0.1, "point_tolerance": 0.11},
#     match_lower_cusps={"value_tolerance": 0.1, "point_tolerance": 0.11}
# )
# plot_reconstruction(plot_points, plot_surface_values, reconstruction_ultra, name="4 extra samples")
# print("sum max abs error:", sum_max_abs_error(plot_surface_values, reconstruction_ultra))
# print("gap weighted error:", gap_weighted_error(plot_surface_values, reconstruction_ultra))

# exit(0)

# print("LEN:", len(samples))
reconstruction = reconstruct_active(
    err_diff_tol=0,
    new_points_count=40,
    sample_radius=0.01,
    dist="normal",
    max_iters=0,
    domains=[(0, 2)],
    sample_points=samples,
    approximation_points=plot_points,
    curves=curves_1,
    interpolant_degree=20,
    delta_inner=0.268,
    delta_outer=0.27,
    removal_epsilon=0.0,
    approx_basis=cheb_basis_shifted(0, 2),
    match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.11},
)
plt.plot(plot_points, np.log10(gap_weighted_error_pointwise(plot_surface_values, reconstruction)), label=f"4 extra sample points")

plt.xlabel("x value")
plt.ylabel("log10 of pointwise gap weighted error")
plt.ylim(-16, 0)
plt.legend()
# plt.title("10000 extra sample points")
print("LEN:", len(samples))
plt.scatter(samples, [0] * len(samples))
plt.show()
plot_reconstruction(plot_points, plot_surface_values, reconstruction, name="4 extra samples")
print("sum max abs error:", sum_max_abs_error(plot_surface_values, reconstruction))
print("gap weighted error:", gap_weighted_error(plot_surface_values, reconstruction))

# plt.plot(plot_points, reconstruction[0, :])
plt.scatter([x[0] for x in plot_points[1:-1]], deriv(reconstruction[:, 1], 2.1/(4000 - 1)), s=0.2)
# plt.plot(plot_points, reconstructed_surfaces[1, :])
plt.scatter([x[0] for x in plot_points[1:-1]], deriv(reconstruction[:, 0], 2.1/(4000 - 1)), s=0.2)
plt.show()

exit(0)

# reconstructed_surfaces = reconstruct_frobenius(samples, plot_points, original_surface_values=sample_surface_values, interpolant_degree=20, approx_basis=cheb_basis_shifted(0, 2), dimension=1)
# plot_reconstruction(plot_points, plot_surface_values, reconstructed_surfaces, name="Frobenius")

# print("sum max abs error:", sum_max_abs_error(plot_surface_values, reconstructed_surfaces))
# print("gap weighted error:", gap_weighted_error(plot_surface_values, reconstructed_surfaces))

reconstructed_surfaces, (esps_full, esps_bottom) = reconstruct_enhanced([(0, 2.4)], samples, plot_points, original_surface_values=sample_surface_values, interpolant_degree=20, delta_inner=0.268, delta_outer=0.270, removal_epsilon=0.0, approx_basis=cheb_basis_shifted(0, 2.4), match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.1})
plot_reconstruction(plot_points, plot_surface_values, reconstructed_surfaces, name="Enhanced")

plot_esp_values_full = np.array([[elem_sym_poly(3, 1)(v), elem_sym_poly(3, 2)(v), elem_sym_poly(3, 3)(v)] for v in plot_surface_values])
plot_esp_values_bottom = np.array([[elem_sym_poly(2, 1)(v), elem_sym_poly(2, 2)(v)] for v in plot_surface_values[:, 0:-1]])
plot_reconstruction(plot_points, plot_esp_values_full, np.array(esps_full), name="Full")
plot_reconstruction(plot_points, plot_esp_values_bottom, np.array(esps_bottom), name="Bottom")

# plt.plot(plot_points, reconstructed_surfaces[0, :])
# plt.scatter(plot_points[1:-1], deriv(reconstructed_surfaces[0, :], 2/(4000 - 1)), s=0.2)
# # plt.plot(plot_points, reconstructed_surfaces[1, :])
# plt.scatter(plot_points[1:-1], deriv(reconstructed_surfaces[1, :], 2/(4000 - 1)), s=0.2)
# plt.show()

print("sum max abs error:", sum_max_abs_error(plot_surface_values, reconstructed_surfaces))
print("gap weighted error:", gap_weighted_error(plot_surface_values, reconstructed_surfaces))
exit(0)

# x_values, original_surfaces, reconstructed_surfaces = reconstruct_enhanced(samples, original_surfaces=np.array([curves_1(x) for x in samples]).T, interpolant_degree=20, delta_inner=0.423, delta_outer=0.433, removal_epsilon=0.0, approx_basis=cheb_basis_shifted(0, 2), dimension=1, match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.1})
# plot_reconstruction(x_values, original_surfaces, reconstructed_surfaces, name="Enhanced")

# print("sum max abs error:", sum_max_abs_error(original_surfaces, reconstructed_surfaces))
# print("gap weighted error:", gap_weighted_error(original_surfaces, reconstructed_surfaces))

# x_values, original_surfaces, reconstructed_surfaces = reconstruct_enhanced(samples, original_surfaces=np.array([curves_4(x) for x in samples]).T, interpolant_degree=20, delta_inner=0.3, delta_outer=0.7, removal_epsilon=0.0, approx_basis=cheb_basis_shifted(0, 2), dimension=1, match_cusps=[np.pi/3])
# plot_reconstruction(x_values, original_surfaces, reconstructed_surfaces, name="Enhanced")

## NOTE: when there is no unmatched cusp, it is far more accurate to use all surfaces to construct ESPs
## NOTE: behaviour for curves_2 is the following:
## for curves_2, setting delta_inner = delta_outer = 0.8 drastically reduces the max error (from ~1e-4 to ~1e-10)
## What seems to be happening is: if delta_inner (and delta_outer) is large enough so that the delta_inner ball around the top matched cusp contains the other two matched cusps, then the error is consistently very small (~1e-10) (except slightly higher only at the matched cusp points, which is to be expected). however, if the delta_inner ball is not large enough to cover both, the matched cusp not contained in the delta_inner ball pollutes the error around it: the error is not only higher at this matched cusp, but also everywhere beyond the delta_inner ball.
## for curves_4, there are no matched cusps arising from the bottom two surfaces, only from the top two surfaces. even here, a larger delta_inner is better (although for some reason there is a large spike in the max abs and gap weighted errors near delta_inner = 1). but this actually makes sense, because if delta_inner is small, since we are globally approximating the bottom two curves, we will have a suddent gradient change either side of the matched cusp, and delta_inner being small means this gradient change happens quickly, so is harder to approximate. a solution could be individually approximating between each pair of matched cusps (though this might be hard to generalise to higher dimensions), or using a larger delta_outer (relative to delta_inner), or keeping delta_inner the same but including less data near the cusp in the approximation (the approximation that uses fewer ESPs)
## we run into the following problem: if we are approximating only using bottom curves outside delta_outer, and only using all curves inside delta_inner, what happens between delta_outer and delta_inner? the sigmoid trick doesn't seem to perform that well. using only all curves is equivalent to making delta_inner = delta_outer (enlarge delta_inner), only using bottom curves is equivalent to making delta_outer = delta_inner (shrink delta_outer)
    ## possible solution to the problem of individual approximation in higher dimensions: partition the convex hull of the matched cusp points into convex hulls (or just polyhedra) of subsets of the points. there is something called a Delaunay triangulation which achieves this (partitions into simplices), but unfortunately this has time complexity O(m^(n/2)) where m is number of points, n is dimension (although apparently in practice it is often faster than this)
    ## another solution (simpler): split into rectangles, new rectangle for each x (or y) coord (generalised in higher dimensions) of matched cusps.
    ## these methods might not work anyway, because we still run into the same problem of having a sudden derivative change, which wouldn't be all the way around the ball, but would be around some of it
    ## if partition idea worked, we _would_ still use ESPs as the location of the matched cusps wouldn't be exact

# print("sum max abs error:", sum_max_abs_error(original_surfaces, reconstructed_surfaces))
# print("gap weighted error:", gap_weighted_error(original_surfaces, reconstructed_surfaces))

# original_surfaces, reconstructed_surfaces = reconstruct_direct(samples, curves=intersecting_cosine_curves_1, interpolant_degree=20, approx_basis=cheb_basis_shifted(0, 2), dimension=1)
# plot_reconstruction(samples, original_surfaces, reconstructed_surfaces, name="Direct")

# plot_convergence(gap_weighted_error, range(10, 45), title="Gap weighted error")
# plot_convergence(sum_max_abs_error, range(10, 45), title="Sum max abs error")
# plot_reconstruction(samples, original_surfaces, reconstructed_surfaces, name="Direct")

# compare with direct interpolation - put plots next to each other and error plots next to each other (or on same plot)
# plot convergence in degree (of interpolant) for direct interpolation, regular frob companion and this method
# think about proving error bounds (e.g. for derivatives)
# try to add a "request" for more data somewhere in algorithm to simulate real life
# try to make it work in 2d
