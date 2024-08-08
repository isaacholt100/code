import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

from bases import cheb_basis
from cusps.active import empirical_distribution_estimate, reconstruct_active, request_random_points
from cusps.errors import gap_weighted_error, gap_weighted_error_pointwise, sum_max_abs_error
from cusps.esps import elem_sym_poly, reconstruct_roots
from cusps.find_cusps import reconstruct_frobenius
from cusps.reconstruct import cheb_basis_shifted, deriv, find_matched_cusp_points, get_surface_values, reconstruct_enhanced
from custom_types import Curves, FloatArray
from polyapprox.analyses import plotting_grid_points

def curves_1(x: float):
    y1 = math.sin(x)
    y2 = math.cos(2*x)
    y3 = math.sin(2*x)
    y4 = 1/3 + math.cos(2*x/3)
    return sorted([y1, y2, y3, y4])[0:3]

def curves_2(x: float):
    return sorted([math.cos(x) + 1, math.sin(x/2) + 1, math.cos(3*x - 0.9) + 0.5])

def curves_3(x: float):
    return sorted([1/2*math.cos(x)+1/2, 1/2*math.sin(2*x) + 1/2, 1/2*math.sin(x)+1.1])

def curves_4(x: float):
    return sorted([math.cos(x) + 1, math.sin(x/2) + 1, math.cos(x - 0.9)])

def curves_sinh(a: float, b: float):
    def curves(x: float, y: float):
        return [math.sinh(-math.sqrt(a**2 * y**2 + b**2 * x**2)/(a*b)), math.sinh(math.sqrt(a**2 * y**2 + b**2 * x**2)/(a*b))]
    return curves

_curves: list[Curves] = [curves_1, curves_2, curves_3, curves_4, curves_sinh(1, 1)]

def rand_unif_samples(a: float, b: float, sample_size: int) -> FloatArray:
    assert sample_size > 0
    return np.sort(np.random.uniform(a, b, sample_size))

samples = np.transpose([rand_unif_samples(0, 2.4, 4001)])

def plot_reconstruction(x_values, original_surfaces, reconstructed_surfaces, name: str):
    plt.figure(0)
    for i in range(original_surfaces.shape[1]):
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
    plt.title(f"Error plot ({name})")
    plt.show()

def plot_params():
    values = np.linspace(0.0, 0.5, 10)
    max_abs_errors, gap_weighted_errors = [], []
    for value in values:
        original_surfaces = get_surface_values(curves_1, samples)
        reconstructed_surfaces, _ = reconstruct_enhanced([(0, 2)], samples, samples, original_surface_values=original_surfaces, interpolant_degree=20, delta_p=0.15, delta_q=0.268+value, removal_epsilon=0.0, approx_basis=cheb_basis_shifted(0, 2), match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.1})
        max_abs_errors.append(sum_max_abs_error(original_surfaces, reconstructed_surfaces))
        gap_weighted_errors.append(gap_weighted_error(original_surfaces, reconstructed_surfaces))

    plt.plot(values, np.log10(max_abs_errors))
    plt.ylabel("max abs error")
    plt.xlabel("delta_p")
    plt.show()
    plt.plot(values, np.log10(gap_weighted_errors))
    plt.ylabel("gap weighted error")
    plt.xlabel("delta_p")
    plt.show()
# plot_params()

plot_points = np.transpose([np.linspace(0.0, 2.4, 4000)])

# TODO: interpolation and quadrature rule for spherical harmonics

sample_surface_values = get_surface_values(curves_1, samples)
plot_surface_values = get_surface_values(curves_1, plot_points)

# TODO: TRY various different distributions e.g. (spikey) normal, triangular, 1/r dist
# make it log plot on y axis
# TODO: try the two different methods that Christoph mentioned (on the sheet of paper)
# TODO: plot errors in the ESPs rather than the surfaces
# for max_iters in [*range(2, 10), 10, 20, 50, 100]:
# empirical_distribution_estimate([100], 1000, 10, 0.2, [(0, 2.4)], samples, plot_points, curves_1, interpolant_degree=20, delta_p=0.20, delta_q=0.27, removal_epsilon=0.0, approx_basis=cheb_basis_shifted(0, 2.4), match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.11})


# empirical_distribution_estimate([10, 20, 30, 40, 50], 1000, 10, 0.2, [(0, 2.4)], samples, plot_points, curves_1, interpolant_degree=20, delta_p=0.20, delta_q=0.27, removal_epsilon=0.0, approx_basis=cheb_basis_shifted(0, 2.4), match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.11})

samples_2d = np.random.uniform(-1, 1, (4001, 2))
plot_points_2d = plotting_grid_points([(-1, 1), (-1, 1)], 4000)

empirical_distribution_estimate([2, 4, 6, 8, 10, 20, 30, 40, 50], 1000, 10, 0.1, [(-1, 1), (-1, 1)], samples_2d, plot_points_2d, curves_sinh(4/3, 12/5), interpolant_degree=20, delta_p=0.20, delta_q=0.27, removal_epsilon=0.0, approx_basis=cheb_basis, match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.11})

exit(0)

# reconstruction = reconstruct_active(
#     err_diff_tol=1e-5,
#     new_points_count=40,
#     sample_radius=0.2,
#     dist="normal",
#     max_iters=1,
#     domains=[(0, 2)],
#     sample_points=samples,
#     approximation_points=plot_points,
#     curves=curves_1,
#     interpolant_degree=20,
#     delta_p=0.268,
#     delta_q=0.270,
#     removal_epsilon=0.0,
#     approx_basis=cheb_basis_shifted(0, 2.4),
#     match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.1}
# )
# print("sum max abs error:", sum_max_abs_error(plot_surface_values, reconstruction))
# print("gap weighted error:", gap_weighted_error(plot_surface_values, reconstruction))

# exit(0)

# reconstructed_surfaces = reconstruct_frobenius(samples, plot_points, original_surface_values=sample_surface_values, interpolant_degree=20, approx_basis=cheb_basis_shifted(0, 2), dimension=1)
# plot_reconstruction(plot_points, plot_surface_values, reconstructed_surfaces, name="Frobenius")

# print("sum max abs error:", sum_max_abs_error(plot_surface_values, reconstructed_surfaces))
# print("gap weighted error:", gap_weighted_error(plot_surface_values, reconstructed_surfaces))

reconstructed_surfaces, (esps_full, esps_bottom) = reconstruct_enhanced([(0, 2.4)], samples, plot_points, original_surface_values=sample_surface_values, interpolant_degree=20, delta_p=0.268, delta_q=0.270, removal_epsilon=0.0, approx_basis=cheb_basis_shifted(0, 2.4), match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.1})
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

# x_values, original_surfaces, reconstructed_surfaces = reconstruct_enhanced(samples, original_surfaces=np.array([curves_1(x) for x in samples]).T, interpolant_degree=20, delta_p=0.423, delta_q=0.433, removal_epsilon=0.0, approx_basis=cheb_basis_shifted(0, 2), dimension=1, match_cusps={"value_tolerance": 0.1, "point_tolerance": 0.1})
# plot_reconstruction(x_values, original_surfaces, reconstructed_surfaces, name="Enhanced")

# print("sum max abs error:", sum_max_abs_error(original_surfaces, reconstructed_surfaces))
# print("gap weighted error:", gap_weighted_error(original_surfaces, reconstructed_surfaces))

# x_values, original_surfaces, reconstructed_surfaces = reconstruct_enhanced(samples, original_surfaces=np.array([curves_4(x) for x in samples]).T, interpolant_degree=20, delta_p=0.3, delta_q=0.7, removal_epsilon=0.0, approx_basis=cheb_basis_shifted(0, 2), dimension=1, match_cusps=[np.pi/3])
# plot_reconstruction(x_values, original_surfaces, reconstructed_surfaces, name="Enhanced")

## NOTE: when there is no unmatched cusp, it is far more accurate to use all surfaces to construct ESPs
## NOTE: behaviour for curves_2 is the following:
## for curves_2, setting delta_p = delta_q = 0.8 drastically reduces the max error (from ~1e-4 to ~1e-10)
## What seems to be happening is: if delta_p (and delta_q) is large enough so that the delta_p ball around the top matched cusp contains the other two matched cusps, then the error is consistently very small (~1e-10) (except slightly higher only at the matched cusp points, which is to be expected). however, if the delta_p ball is not large enough to cover both, the matched cusp not contained in the delta_p ball pollutes the error around it: the error is not only higher at this matched cusp, but also everywhere beyond the delta_p ball.
## for curves_4, there are no matched cusps arising from the bottom two surfaces, only from the top two surfaces. even here, a larger delta_p is better (although for some reason there is a large spike in the max abs and gap weighted errors near delta_p = 1). but this actually makes sense, because if delta_p is small, since we are globally approximating the bottom two curves, we will have a suddent gradient change either side of the matched cusp, and delta_p being small means this gradient change happens quickly, so is harder to approximate. a solution could be individually approximating between each pair of matched cusps (though this might be hard to generalise to higher dimensions), or using a larger delta_q (relative to delta_p), or keeping delta_p the same but including less data near the cusp in the approximation (the approximation that uses fewer ESPs)
## we run into the following problem: if we are approximating only using bottom curves outside delta_q, and only using all curves inside delta_p, what happens between delta_q and delta_p? the sigmoid trick doesn't seem to perform that well. using only all curves is equivalent to making delta_p = delta_q (enlarge delta_p), only using bottom curves is equivalent to making delta_q = delta_p (shrink delta_q)
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

def Hamiltonian_Graphene(kx,ky):
    a = 2.46 # Wikipedia
    t = 1
    k = np.array((kx,ky))
    d1 = (a/2)*np.array((1,np.sqrt(3)))
    d2 = (a/2)*np.array((1,-np.sqrt(3)))
    d3 = -a*np.array((1,0))
    sx = np.matrix([[0,1],[1,0]]);
    sy = np.matrix([[0,-1j],[1j,0]]);
    hx = np.cos(k@d1)+np.cos(k@d2)+np.cos(k@d3)
    hy = np.sin(k@d1)+np.sin(k@d2)+np.sin(k@d3)
    H = -t*(hx*sx - hy*sy)
    return H

def plot_dispersion(m,n,BZx1,BZx2,BZy1,BZy2):
    # Generate a mesh
    kx_range = np.linspace(BZx1, BZx2, num=m)
    ky_range = np.linspace(BZy1, BZy2, num=n)
    # Get the number of levels with a dummy call (an NxN square matrix has N levels)
    num_levels = len(Hamiltonian_Graphene(1,1))
    energies = np.zeros((m,n,num_levels)); # initialize
    # Now iterate over discretized mesh, to consider each coordinate.
    X = []
    Y = []
    for i in range(m):
        for j in range(n):
            H = Hamiltonian_Graphene(kx_range[i],ky_range[j]);
            evals, evecs = np.linalg.eigh(H); # Numerically get eigenvalues and eigenvectors
            energies[i, j, :]=evals;
            X.append(kx_range[i])
            Y.append(ky_range[j])
    return kx_range, ky_range, energies

# Run it!
BZx1=-1;BZx2=+1;BZy1=-1;BZy2=+1;
M = 71
N = 71
X, Y, energiesC = plot_dispersion(M,N,BZx1,BZx2,BZy1,BZy2)
x1, y1 = np.meshgrid(X, Y, indexing="ij")
input_values = np.vstack((x1.flatten(), y1.flatten())).T

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
energies1 = np.array([f3(x, y).T for (x, y) in input_values])
energies2 = np.array([f4(x, y).T for (x, y) in input_values])
energies3 = np.array([f1(x, y).T for (x, y) in input_values])
energies4 = np.array([f2(x, y).T for (x, y) in input_values])
energies1, energies2 = energiesC[:,:,0].flatten(), energiesC[:,:,1].flatten()

def g1(x, y):
    return np.cos(4*x) - np.cos(4*y) - 4
def g2(x, y):
    return 2.5
energies3 = np.array([g1(x, y) for (x, y) in input_values])
energies4 = np.array([g2(x, y) for (x, y) in input_values])
sorted_curves = np.partition(np.vstack([energies1, energies2, energies3, energies4]), [0, 1, 2, 3], axis=0)

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.contour3D(x1, y1, sorted_curves[0].reshape(x1.shape), 100, cmap="binary")
# ax.contour3D(x1, y1, sorted_curves[1].reshape(x1.shape), 100, cmap="plasma")
# ax.contour3D(x1, y1, sorted_curves[2].reshape(x1.shape), 100, cmap="viridis")
# ax.contour3D(x1, y1, sorted_curves[3].reshape(x1.shape), 100, cmap="binary")
# # ax.view_init(30, 35)
# plt.show()

def compare_delaunay(input_values, delta_p: float, delta_q: float):
    original_surfaces = np.array(sorted_curves[0:-1])
    reconstructed_surfaces_delaunay, fn = reconstruct_enhanced([(-1, 1), (-1, 1)], input_values, input_values, np.array(sorted_curves[0:-1]), interpolant_degree=20, delta_p=delta_p, delta_q=delta_q, removal_epsilon=0.0, approx_basis=cheb_basis, match_cusps={"value_tolerance": 0.2, "point_tolerance": 0.1}, use_delaunay=True)
    log_errors_delaunay = np.log10(np.sum([np.abs(reconstructed_surfaces_delaunay[i, :] - original_surfaces[i, :]) for i in range(reconstructed_surfaces_delaunay.shape[0] - 1)], axis=0))
    print(sum_max_abs_error(original_surfaces, reconstructed_surfaces_delaunay), gap_weighted_error(original_surfaces, reconstructed_surfaces_delaunay))

    plt.scatter(*zip(*input_values), c=log_errors_delaunay, cmap="viridis")
    fn()
    plt.colorbar()
    plt.clim(-8, 0)
    plt.show()

    reconstructed_surfaces_enhanced, _ = reconstruct_enhanced([(-1, 1), (-1, 1)], input_values, input_values, np.array(sorted_curves[0:-1]), interpolant_degree=20, delta_p=delta_p, delta_q=delta_q, removal_epsilon=0.0, approx_basis=cheb_basis, match_cusps={"value_tolerance": 0.2, "point_tolerance": 0.1}, use_delaunay=False)
    log_errors_enhanced = np.log10(np.sum([np.abs(reconstructed_surfaces_enhanced[i, :] - original_surfaces[i, :]) for i in range(reconstructed_surfaces_enhanced.shape[0] - 1)], axis=0))
    print(sum_max_abs_error(original_surfaces, reconstructed_surfaces_enhanced), gap_weighted_error(original_surfaces, reconstructed_surfaces_enhanced))

    plt.scatter(*zip(*input_values), c=log_errors_enhanced, cmap="viridis")
    plt.colorbar()
    plt.clim(-8, 0)
    plt.show()

    reconstructed_surfaces_frobenius = reconstruct_frobenius(input_values, input_values, np.array(sorted_curves[0:-1]), interpolant_degree=20, approx_basis=cheb_basis, dimension=2)

    log_errors_frobenius = np.log10(np.sum([np.abs(reconstructed_surfaces_frobenius[i, :] - original_surfaces[i, :]) for i in range(reconstructed_surfaces_frobenius.shape[0] - 1)], axis=0))
    print(sum_max_abs_error(original_surfaces, reconstructed_surfaces_frobenius), gap_weighted_error(original_surfaces, reconstructed_surfaces_frobenius))

    # cax1 = axs[0].imshow(log_errors_delaunay, vmin=vmin, vmax=vmax, cmap="viridis")
    # cax2 = axs[1].imshow(log_errors_frobenius, vmin=vmin, vmax=vmax, cmap="viridis")
    # cax3 = axs[2].imshow(log_errors_enhanced, vmin=vmin, vmax=vmax, cmap="viridis")

    # cbar = fig.colorbar(cax1, ax=axs, orientation="vertical", fraction=0.02, pad=0.04)
    # cbar.set_label("log10 of absolute error")
    # plt.show()

compare_delaunay(input_values, delta_p=0.05, delta_q=0.06)

def plot_params_2d():
    values = np.linspace(0.3, 0.5, 10)
    max_abs_errors, gap_weighted_errors = [], []
    matched_cusp_points = find_matched_cusp_points(input_values, np.array(sorted_curves[0:-1]), interpolant_degree=20, value_tolerance=0.2, point_tolerance=0.1, approx_basis=two_cheb_basis, dimension=2)
    print(matched_cusp_points)
    for value in values:
        print("progress:", value)
        original_surfaces = np.array(sorted_curves[0:-1])
        reconstructed_surfaces, _ = reconstruct_enhanced([(-1, 1), (-1, 1)], input_values, input_values, np.array(sorted_curves[0:-1]), interpolant_degree=20, delta_p=value, delta_q=value+0.01, removal_epsilon=0.00, approx_basis=two_cheb_basis, match_cusps=matched_cusp_points, use_delaunay=False)
        max_abs_errors.append(sum_max_abs_error(original_surfaces, reconstructed_surfaces))
        gap_weighted_errors.append(gap_weighted_error(original_surfaces, reconstructed_surfaces))

    plt.plot(values, np.log10(max_abs_errors))
    plt.ylabel("max abs error")
    plt.xlabel("delta_p")
    plt.show()
    plt.plot(values, np.log10(gap_weighted_errors))
    plt.ylabel("gap weighted error")
    plt.xlabel("delta_p")
    plt.show()

# plot_params_2d()

## TODO: what distribution of points is best to approximate around matched cusp (shoudl be clustered around the cusps)
## TODO: Schmeisser matrix learning