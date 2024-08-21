import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
#from scipy.optimize import basinhopping
#import time
import scipy as sp
import pyvista as pv
import numpy as np
import random

from cusps.reconstruct import get_surface_values
from plot_2d import plot_2d
from polyapprox.analyses import plotting_grid_points
np.bool = np.bool_

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
            evals, evecs = LA.eigh(H); # Numerically get eigenvalues and eigenvectors
            energies[i,j,:]=evals;
            X.append(kx_range[i])
            Y.append(ky_range[j])
    return kx_range, ky_range, energies

def g1(x, y):
    return math.sin(4*x) - math.sin(4*y) - 2.5
def g2(x, y):
    return 2.5

def graphene_surfaces(x: float, y: float):
    H = Hamiltonian_Graphene(x, y)
    evals, _ = np.linalg.eigh(H)
    return sorted(evals.tolist() + [g1(x, y), g2(x, y)])[0:-1]

# Run it!
M = 150
N = 150
L = 100

np.random.seed(35) # make things reproducible

input_values = plotting_grid_points([(-1, 1), (-1, 1)], 150**2)
x1, y1 = input_values[:, 0].reshape((150, 150)), input_values[:, 1].reshape((150, 150))
energiesC = get_surface_values(graphene_surfaces, input_values)

plot_2d(input_values, energiesC)