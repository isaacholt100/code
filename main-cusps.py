import matplotlib.pyplot as plt
import numpy as np

from cusps.errors import gap_weighted_error, sum_max_abs_error
from cusps.reconstruct import cheb_basis_shifted, intersecting_cosine_curves_1, reconstruct_direct, reconstruct_enhanced, reconstruct_frobenius, two_cheb_basis

def rand_unif_samples(a: float, b: float, sample_size: int):
    assert sample_size > 0
    return np.sort(np.random.uniform(a, b, sample_size))

samples = rand_unif_samples(0, 2, 4000)

def plot_reconstruction(x_values, original_surfaces, reconstructed_surfaces, name: str):
    plt.figure(0)
    plt.plot(x_values, original_surfaces[0, :])
    plt.plot(x_values, original_surfaces[1, :])
    plt.plot(x_values, original_surfaces[2, :])
    plt.title("Original multi-surface data")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

    plt.figure(1)
    plt.plot(x_values, reconstructed_surfaces[0, :])
    plt.plot(x_values, reconstructed_surfaces[1, :])
    plt.plot(x_values, reconstructed_surfaces[2, :])
    plt.title(f"ESP-reconstructed multi-surface ({name})")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

    errors = [np.abs(reconstructed_surfaces[i, :] - original_surfaces[i, :]) for i in range(reconstructed_surfaces.shape[0] - 1)]
    for error_points in errors:
        plt.plot(x_values, np.log10(error_points))
    plt.ylabel("log10 of error")
    plt.title(f"Error plot ({name})")
    plt.show()

def plot_params():
    values = np.linspace(0.01, 1, 50)
    max_abs_errors, gap_weighted_errors = [], []
    for value in values:
        _, original_surfaces, reconstructed_surfaces = reconstruct_enhanced(0, 2, samples, original_surfaces=np.array([intersecting_cosine_curves_1(x) for x in samples]).T, interpolant_degree=20, smoothing_degree=0, delta_p=value, removal_epsilon=0.0, smoothing_sample_range=0.1, approx_basis=cheb_basis_shifted(0, 2), dimension=1, matched_cusp_point_search_tolerance=0.1)
        max_abs_errors.append(sum_max_abs_error(original_surfaces, reconstructed_surfaces))
        gap_weighted_errors.append(gap_weighted_error(original_surfaces, reconstructed_surfaces))

    plt.plot(values, max_abs_errors)
    plt.ylabel("max abs error")
    plt.xlabel("delta_p")
    plt.show()
    plt.plot(values, gap_weighted_errors)
    plt.ylabel("gap weighted error")
    plt.xlabel("delta_p")
    plt.show()
# plot_params()

x_values, original_surfaces, reconstructed_surfaces = reconstruct_enhanced(0, 2, samples, original_surfaces=np.array([intersecting_cosine_curves_1(x) for x in samples]).T, interpolant_degree=20, smoothing_degree=0, delta_p=0.272, removal_epsilon=0.000, smoothing_sample_range=0.1, approx_basis=cheb_basis_shifted(0, 2), dimension=1, matched_cusp_point_search_tolerance=0.1)
plot_reconstruction(x_values, original_surfaces, reconstructed_surfaces, name="Enhanced")

print("sum max abs error:", sum_max_abs_error(original_surfaces, reconstructed_surfaces))
print("gap weighted error:", gap_weighted_error(original_surfaces, reconstructed_surfaces))

# original_surfaces, reconstructed_surfaces = reconstruct_frobenius(samples, curves=intersecting_cosine_curves_1, interpolant_degree=20, approx_basis=cheb_basis_shifted(0, 2), dimension=1)
# plot_reconstruction(samples, original_surfaces, reconstructed_surfaces, name="Frobenius")

# original_surfaces, reconstructed_surfaces = reconstruct_direct(samples, curves=intersecting_cosine_curves_1, interpolant_degree=20, approx_basis=cheb_basis_shifted(0, 2), dimension=1)
# plot_reconstruction(samples, original_surfaces, reconstructed_surfaces, name="Direct")

def plot_convergence(error_fn, degrees, title: str):
    errors_enhanced = []
    errors_frobenius = []
    errors_direct = []
    for degree in degrees:
        _, original_surfaces, reconstructed_surfaces = reconstruct_enhanced(0, 2, samples, original_surfaces=np.array([intersecting_cosine_curves_1(x) for x in samples]).T, interpolant_degree=degree, smoothing_degree=8, delta_p=0.211, removal_epsilon=0.618, smoothing_sample_range=0.15, approx_basis=cheb_basis_shifted(0, 2), dimension=1, matched_cusp_point_search_tolerance=0.1)
        errors_enhanced.append(error_fn(original_surfaces, reconstructed_surfaces))

        original_surfaces, reconstructed_surfaces = reconstruct_frobenius(samples, curves=intersecting_cosine_curves_1, interpolant_degree=degree, approx_basis=cheb_basis_shifted(0, 2), dimension=1)
        errors_frobenius.append(error_fn(original_surfaces, reconstructed_surfaces))

        original_surfaces, reconstructed_surfaces = reconstruct_direct(samples, curves=intersecting_cosine_curves_1, interpolant_degree=degree, approx_basis=cheb_basis_shifted(0, 2), dimension=1)
        errors_direct.append(error_fn(original_surfaces, reconstructed_surfaces))
    
    plt.plot(degrees, errors_enhanced, label="Enhanced")
    plt.plot(degrees, errors_frobenius, label="Frobenius")
    plt.plot(degrees, errors_direct, label="Direct")
    plt.xlabel("Interpolant degree")
    plt.ylabel("Error")
    plt.title(title)
    plt.legend()
    plt.show()

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
N = 100
M = 120
X, Y, energiesC = plot_dispersion(N,M,BZx1,BZx2,BZy1,BZy2)
x1, y1 = np.meshgrid(X, Y)
input_values = np.vstack((x1.flatten(), y1.flatten())).T

energies1, energies2 = energiesC[:,:,0].flatten(), energiesC[:,:,1].flatten()
a1, b1 = 2, 3
a2, b2 = 3, 4
energies3 = np.array([np.sinh(np.sqrt(a1**2 * y**2 + b1**2 * x**2)/(a1 * b1)).T for (x, y) in input_values])
energies4 = np.array([np.sinh(np.sqrt(a2**2 * y**2 + b2**2 * x**2)/(a2 * b2)).T for (x, y) in input_values])
sorted_curves = np.partition(np.vstack([energies1, energies2, energies3, energies4]), [0, 1, 2, 3], axis=0)

# input_values, original_surfaces, reconstructed_surfaces = reconstruct_enhanced(-1, 1, input_values, np.array(sorted_curves[0:-1]), interpolant_degree=20, smoothing_degree=0, delta_p=0.121, removal_epsilon=0.0, smoothing_sample_range=0.1, approx_basis=two_cheb_basis, dimension=2, matched_cusp_point_search_tolerance=0.005)

# errors = [np.abs(reconstructed_surfaces[i, :] - original_surfaces[i, :]) for i in range(reconstructed_surfaces.shape[0] - 1)]
# plt.scatter(*zip(*input_values), c=errors[0], cmap="viridis")
# plt.colorbar()
# plt.show()
# print(sum_max_abs_error(original_surfaces, reconstructed_surfaces))

def plot_params_2d():
    values = np.linspace(0.01, 1, 30)
    max_abs_errors, gap_weighted_errors = [], []
    for value in values:
        _, original_surfaces, reconstructed_surfaces = _, original_surfaces, reconstructed_surfaces = reconstruct_enhanced(-1, 1, input_values, np.array(sorted_curves[0:-1]), interpolant_degree=20, smoothing_degree=0, delta_p=value, removal_epsilon=0, smoothing_sample_range=0.1, approx_basis=two_cheb_basis, dimension=2, matched_cusp_point_search_tolerance=0.1)
        max_abs_errors.append(sum_max_abs_error(original_surfaces, reconstructed_surfaces))
        gap_weighted_errors.append(gap_weighted_error(original_surfaces, reconstructed_surfaces))

    plt.plot(values, max_abs_errors)
    plt.ylabel("max abs error")
    plt.xlabel("delta_p")
    plt.show()
    plt.plot(values, gap_weighted_errors)
    plt.ylabel("gap weighted error")
    plt.xlabel("delta_p")
    plt.show()

# plot_params_2d()