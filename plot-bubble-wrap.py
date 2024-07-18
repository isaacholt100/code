from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv

from cusps.transition import matched_cusps_approx_weight

x_range = (-3, 3)
y_range = (-3, 3)

x_plot, y_plot = np.meshgrid(np.linspace(*x_range, 200), np.linspace(*y_range, 200), indexing="ij")
input_values = np.vstack((x_plot.flatten(), y_plot.flatten())).T
scale_factor = 2 # change this to adjust how close the matched cusp points are. with delta_p = 0.2, delta_q = 0.8, sf~0.8 gives overlapping "bubbles", sf~2 gives nicely spaced "bubbles"
matched_cusp_points = list(map(lambda l: np.array(l)*scale_factor, [[0, -1], [0, 1], [np.sqrt(3)/2, 1/2], [np.sqrt(3)/2, -1/2], [-np.sqrt(3)/2, 1/2], [-np.sqrt(3)/2, -1/2]])) # scaled points on a hexagon, similar to graphene matched cusps
z_plot = [matched_cusps_approx_weight(delta_p=0.2, delta_q=0.8, matched_cusp_points=matched_cusp_points, x=x) for x in input_values]
z_plot = np.array(z_plot).reshape(x_plot.shape)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# The pyvista plot starts here
pl = pv.Plotter()
scalefactor = np.abs(np.max(z_plot)) # make axes look like similar size to avoid strange shape
mesh1 = pv.StructuredGrid(x_plot, y_plot, z_plot)

pl.add_mesh(mesh1,opacity=1,color=colors[0],show_edges=False,edge_color=colors[0],point_size=15)

pl.show_grid(axes_ranges=[*x_range, *y_range, np.min(z_plot), np.max(z_plot)],ztitle='z', xtitle='x', ytitle='y',grid='back', all_edges=False,padding=0.1,show_zlabels=True,n_xlabels=5,n_ylabels=5,n_zlabels=3,use_3d_text=False)
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