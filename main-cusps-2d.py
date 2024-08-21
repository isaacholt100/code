import math
from matplotlib import pyplot as plt
import numpy as np

from bases import cheb_basis
from cusps.active import reconstruct_active
from cusps.errors import gap_weighted_error, gap_weighted_error_pointwise, sum_abs_error_pointwise, sum_max_abs_error
from cusps.find_cusps import reconstruct_frobenius
from cusps.reconstruct import get_surface_values, reconstruct_enhanced, reconstruct_ultra
from cusps.surfaces import sample_values_with_noise, sinusoid_surfaces
from custom_types import Curves
from cusps.surfaces import graphene_surfaces
from plot_2d import plot_2d
from polyapprox.analyses import plotting_grid_points


a1, b1 = 5, 8
a2, b2 = 3, 4
def f1(x, y):
    return np.sinh(np.sqrt(a1**2 * y**2 + b1**2 * x**2)/(a1 * b1))
def f2(x, y):
    return np.sinh(np.sqrt(a2**2 * y**2 + b2**2 * x**2)/(a2 * b2))
def f3(x, y): 
    return 3*np.sin(np.sqrt(4*x**2 + 4*y**2))
def f4(x, y):
    return -3*np.cos(np.sqrt(3*x**2 + 5*y**2))

sample_points = plotting_grid_points([(-1, 1), (-1, 1)], 50**2)
surface_values = get_surface_values(graphene_surfaces, sample_points)

plot_points = plotting_grid_points([(-1, 1), (-1, 1)], 76**2)
plot_values = get_surface_values(graphene_surfaces, plot_points)


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
            approx_basis=cheb_basis
        )
        reconstruction_enhanced, _ = reconstruct_enhanced(
            domains=[(-1, 1), (-1, 1)],
            sample_points=sample_points,
            approximation_points=plot_points,
            original_surface_values=noisy_samples,
            interpolant_degree=20,
            removal_epsilon=0.0,
            delta_inner=0.49,
            delta_outer=0.70,
            approx_basis=cheb_basis,
            match_cusps={"value_tolerance": 0.2, "point_tolerance": 0.1},
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

plot_error_against_noise(sample_points, plot_points, graphene_surfaces, np.linspace(0, 0.01, 30))

# exit(0)

plot_2d(plot_points, plot_values)
# TODO: add noise (normal dist, increase stddev until it breaks down)
# TRY MAE error as well
# prepare a 15 min demo of the method. compare the one in the old paper to the new one, show with some toy problems. problems with old method, how they are fixed with new method. could start with the 1d problem, graphene, sinusoid 2d surfaces. can do slides or document
# reconstruction_ultra = reconstruct_ultra(
#     domains=[(-1, 1), (-1, 1)],
#     sample_points=input_values,
#     approximation_points=plot_points,
#     sample_values=surface_values,
#     interpolant_degree=20,
#     delta_top=0.2,
#     delta_lower=0.2,
#     delta_away_from=0.18,
#     approx_basis=cheb_basis,
#     match_top_cusps={"value_tolerance": 0.2, "point_tolerance": 0.1},
#     match_lower_cusps={"value_tolerance": 0.2, "point_tolerance": 0.1}
# )

# delta_inner = 0.59, delta_outer = 0.60 are (close to) optimal for graphene_surfaces
# delta_inner = 0.29, delta_outer = 0.30 are (close to) optimal for sinusoid_surfaces
noisy_surface_values = sample_values_with_noise(0.005, surface_values)
reconstruction_enhanced, _ = reconstruct_enhanced([(-1, 1), (-1, 1)], sample_points, plot_points, noisy_surface_values, interpolant_degree=20, delta_inner=0.49, delta_outer=0.7, removal_epsilon=0.0, approx_basis=cheb_basis, match_cusps={"value_tolerance": 0.2, "point_tolerance": 0.1}, use_delaunay=False)

plot_2d(plot_points, reconstruction_enhanced)

print(sum_max_abs_error(plot_values, reconstruction_enhanced), gap_weighted_error(plot_values, reconstruction_enhanced))

plt.scatter(*zip(*plot_points), c=np.log10(sum_abs_error_pointwise(plot_values, reconstruction_enhanced)), cmap="viridis")
plt.colorbar()
plt.clim(-8, 0)
plt.title("Max abs error plot")
plt.show()

plt.scatter(*zip(*plot_points), c=np.log10(gap_weighted_error_pointwise(plot_values, reconstruction_enhanced)), cmap="viridis")
plt.colorbar()
plt.clim(-8, 0)
plt.title("Gap weighted error plot")
plt.show()

reconstruction_frobenius = reconstruct_frobenius(
    sample_points,
    plot_points,
    noisy_surface_values,
    interpolant_degree=20,
    approx_basis=cheb_basis
)
reconstruction_frobenius = np.delete(reconstruction_frobenius, -1, axis=1)
plot_2d(plot_points, reconstruction_frobenius)
print(sum_max_abs_error(plot_values, reconstruction_frobenius), gap_weighted_error(plot_values, reconstruction_frobenius))

plt.scatter(*zip(*plot_points), c=np.log10(sum_abs_error_pointwise(plot_values, reconstruction_frobenius)), cmap="viridis")
plt.colorbar()
plt.clim(-8, 0)
plt.title("Max abs error plot")
plt.show()

plt.scatter(*zip(*plot_points), c=np.log10(gap_weighted_error_pointwise(plot_values, reconstruction_frobenius)), cmap="viridis")
plt.colorbar()
plt.clim(-8, 0)
plt.title("Gap weighted error plot")
plt.show()


exit(0)
reconstruction_active = reconstruct_active(
    err_diff_tol=0,
    new_points_count=500,
    sample_radius=0.05,
    dist="normal",
    max_iters=1,
    domains=[(-1, 1), (-1, 1)],
    sample_points=sample_points,
    approximation_points=plot_points,
    curves=sinusoid_surfaces,
    interpolant_degree=20,
    delta_inner=0.1,
    delta_outer=0.15,
    removal_epsilon=0.0,
    approx_basis=cheb_basis,
    match_cusps={"value_tolerance": 0.2, "point_tolerance": 0.1}
)
print(sum_max_abs_error(get_surface_values(sinusoid_surfaces, plot_points), reconstruction_active), gap_weighted_error(get_surface_values(sinusoid_surfaces, plot_points), reconstruction_active))

plt.scatter(*zip(*plot_points), c=np.log10(gap_weighted_error_pointwise(get_surface_values(sinusoid_surfaces, plot_points), reconstruction_active)), cmap="viridis")
plt.colorbar()
plt.clim(-8, 0)
plt.show()

exit(0)

def compare_delaunay(input_values, delta_inner: float, delta_outer: float):
    original_surfaces = np.array(sorted_curves[0:-1]).T
    reconstructed_surfaces_delaunay, fn = reconstruct_enhanced([(-1, 1), (-1, 1)], input_values, input_values, np.array(sorted_curves[0:-1]).T, interpolant_degree=20, delta_inner=delta_inner, delta_outer=delta_outer, removal_epsilon=0.0, approx_basis=cheb_basis, match_cusps={"value_tolerance": 0.2, "point_tolerance": 0.1}, use_delaunay=True)
    log_errors_delaunay = np.log10(sum_abs_error_pointwise(original_surfaces, reconstructed_surfaces_delaunay))
    print(sum_max_abs_error(original_surfaces, reconstructed_surfaces_delaunay), gap_weighted_error(original_surfaces, reconstructed_surfaces_delaunay))

    plt.scatter(*zip(*input_values), c=log_errors_delaunay, cmap="viridis")
    fn()
    plt.colorbar()
    plt.clim(-8, 0)
    plt.show()

    reconstructed_surfaces_enhanced, _ = reconstruct_enhanced([(-1, 1), (-1, 1)], input_values, input_values, original_surfaces, interpolant_degree=20, delta_inner=delta_inner, delta_outer=delta_outer, removal_epsilon=0.0, approx_basis=cheb_basis, match_cusps={"value_tolerance": 0.2, "point_tolerance": 0.1}, use_delaunay=False)
    log_errors_enhanced = np.log10(np.sum([np.abs(reconstructed_surfaces_enhanced[i, :] - original_surfaces[i, :]) for i in range(reconstructed_surfaces_enhanced.shape[0] - 1)], axis=0))
    print(sum_max_abs_error(original_surfaces, reconstructed_surfaces_enhanced), gap_weighted_error(original_surfaces, reconstructed_surfaces_enhanced))

    plt.scatter(*zip(*input_values), c=log_errors_enhanced, cmap="viridis")
    plt.colorbar()
    plt.clim(-8, 0)
    plt.show()

    reconstructed_surfaces_frobenius = reconstruct_frobenius(input_values, input_values, original_surfaces, interpolant_degree=20, approx_basis=cheb_basis)

    log_errors_frobenius = np.log10(np.sum([np.abs(reconstructed_surfaces_frobenius[i, :] - original_surfaces[i, :]) for i in range(reconstructed_surfaces_frobenius.shape[0] - 1)], axis=0))
    print(sum_max_abs_error(original_surfaces, reconstructed_surfaces_frobenius), gap_weighted_error(original_surfaces, reconstructed_surfaces_frobenius))

    # cax1 = axs[0].imshow(log_errors_delaunay, vmin=vmin, vmax=vmax, cmap="viridis")
    # cax2 = axs[1].imshow(log_errors_frobenius, vmin=vmin, vmax=vmax, cmap="viridis")
    # cax3 = axs[2].imshow(log_errors_enhanced, vmin=vmin, vmax=vmax, cmap="viridis")

    # cbar = fig.colorbar(cax1, ax=axs, orientation="vertical", fraction=0.02, pad=0.04)
    # cbar.set_label("log10 of absolute error")
    # plt.show()

compare_delaunay(sample_points, delta_inner=0.05, delta_outer=0.06)

# def plot_params_2d():
#     values = np.linspace(0.3, 0.5, 10)
#     max_abs_errors, gap_weighted_errors = [], []
#     matched_cusp_points = find_matched_cusp_points(input_values, np.array(sorted_curves[0:-1]), interpolant_degree=20, value_tolerance=0.2, point_tolerance=0.1, approx_basis=two_cheb_basis, dimension=2)
#     print(matched_cusp_points)
#     for value in values:
#         print("progress:", value)
#         original_surfaces = np.array(sorted_curves[0:-1])
#         reconstructed_surfaces, _ = reconstruct_enhanced([(-1, 1), (-1, 1)], input_values, input_values, np.array(sorted_curves[0:-1]), interpolant_degree=20, delta_inner=value, delta_outer=value+0.01, removal_epsilon=0.00, approx_basis=two_cheb_basis, match_cusps=matched_cusp_points, use_delaunay=False)
#         max_abs_errors.append(sum_max_abs_error(original_surfaces, reconstructed_surfaces))
#         gap_weighted_errors.append(gap_weighted_error(original_surfaces, reconstructed_surfaces))

#     plt.plot(values, np.log10(max_abs_errors))
#     plt.ylabel("max abs error")
#     plt.xlabel("delta_inner")
#     plt.show()
#     plt.plot(values, np.log10(gap_weighted_errors))
#     plt.ylabel("gap weighted error")
#     plt.xlabel("delta_inner")
#     plt.show()

# plot_params_2d()

## TODO: what distribution of points is best to approximate around matched cusp (shoudl be clustered around the cusps)
## TODO: Schmeisser matrix learning