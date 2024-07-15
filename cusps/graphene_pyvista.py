import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
#from scipy.optimize import basinhopping
#import time
import scipy as sp
import pyvista as pv
import numpy as np
import random
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

# Run it!
M = 150
N = 150
L = 100

np.random.seed(35) # make things reproducible

# Run it!
BZx1=-1;BZx2=+1;BZy1=-1;BZy2=+1;
X, Y, energiesC = plot_dispersion(N,M,BZx1,BZx2,BZy1,BZy2)
energies1, energies2 = energiesC[:,:,0], energiesC[:,:,1]
x1, y1 = np.meshgrid(X, Y, indexing="ij")

# We need the color wheel from matplotlib or else it will look bad
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# The pyvista plot starts here
pl = pv.Plotter()
scalefactor = np.abs(np.max([energies1,energies2])) # make axes look like similar size to avoid strange shape
mesh1 = pv.StructuredGrid(x1, y1, energies1/scalefactor)
mesh2 = pv.StructuredGrid(x1, y1, energies2/scalefactor)

pl.add_mesh(mesh1,opacity=1,color=colors[0],show_edges=True,edge_color=colors[0],point_size=15)
pl.add_mesh(mesh2,opacity=0.99,color=colors[1],show_edges=True,edge_color=colors[1],point_size=15)

pl.show_grid(axes_ranges=[-1,1,-1,1,np.min([energies1,energies2]),np.max([energies1,energies2])],ztitle='z', xtitle='x', ytitle='y',grid='back', all_edges=False,padding=0.1,show_zlabels=True,n_xlabels=5,n_ylabels=5,n_zlabels=4,use_3d_text=False)
pl.show(auto_close=False)

# If you need to save a figure, you have to manually adjust the camera angle with features like these.
# You will also have to add something like font_size=70 to the pl.show_grid function keywords since the font will appear too small on the saved figure.
# Uncomment and play with the following to save a figure (good luck!)
#pl.camera.position = (5.0,6.0,6.0)
#pl.camera.elevation = -20
#pl.camera.azimuth -= 100
#pl.window_size = 3000, 3000
pl.save_graphic("pyvista_plot.pdf")

pl.close()
