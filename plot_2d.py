import math

from matplotlib import pyplot as plt
import numpy as np
from custom_types import FloatArray
import pyvista as pv

def plot_2d(points: FloatArray, values: FloatArray):
    assert points.ndim == 2
    assert values.ndim == 2
    assert values.shape[0] == points.shape[0]
    assert points.shape[1] == 2
    square_size = round(math.sqrt(points.shape[0]))
    assert square_size**2 == points.shape[0]
    x1, y1 = points[:, 0].reshape((square_size, square_size)), points[:, 1].reshape((square_size, square_size))
    plot_surfaces = [values[:, i].reshape(x1.shape) for i in range(values.shape[1])]

    # We need the color wheel from matplotlib or else it will look bad
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # The pyvista plot starts here
    pl = pv.Plotter()
    scalefactor = 1#np.max(np.abs(values)) # make axes look like similar size to avoid strange shape
    for i, plot_surface in enumerate(plot_surfaces):
        mesh = pv.StructuredGrid(x1, y1, plot_surface/scalefactor)
        pl.add_mesh(mesh,opacity=1,color=colors[i],show_edges=True,edge_color=colors[i],point_size=15)

    pl.show_grid(axes_ranges=[-1,1,-1,1,np.min(values),np.max(values)],ztitle='z', xtitle='x', ytitle='y',grid='back', all_edges=False,padding=0.1,show_zlabels=True,n_xlabels=5,n_ylabels=5,n_zlabels=4,use_3d_text=False)
    pl.show(auto_close=False)

    pl.close()