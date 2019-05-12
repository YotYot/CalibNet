import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import numpy as np

import pickle

def get_demo_image():
    from matplotlib.cbook import get_sample_data
    import numpy as np
    f = get_sample_data("axes_grid/bivariate_normal.npy", asfileobj=False)
    z = np.load(f)
    # z is a numpy array of 15x15
    return z, (-3,4,-4,3)

pickle_file = '/home/yotamg/PycharmProjects/PSMNet/plot_data.pickle'
with open(pickle_file, 'rb') as f:
    [all_depth, stereo_err, all_depth_cnt_l, mono_err] = pickle.load(f, encoding='latin1')


fig, ax = plt.subplots() # create a new figure with a default 111 subplot



# fig, ax = plt.subplots(figsize=[5,4])

ax.plot(all_depth[5:], stereo_err[5:] / all_depth_cnt_l[5:], label="Stereo")
ax.plot(all_depth[5:], mono_err[5:] / all_depth_cnt_l[5:], label="Mono")
ax.legend()
ax.set_title('Relative L1 Error for Depth')
ax.set_xlabel('Depth (log, cm)')
ax.set_ylabel('Error (log)')
# ax.set_ylim(0,1)
ax.set_yscale('log')
ax.set_xscale('log')
# plt.show()

plt.draw()
plt.show()

fig, ax = plt.subplots() # create a new figure with a default 111 subplot
ax.plot(all_depth[5:], stereo_err[5:] / all_depth_cnt_l[5:], label="Stereo")
ax.plot(all_depth[5:], mono_err[5:] / all_depth_cnt_l[5:], label="Mono")
ax.legend()
ax.set_ylim(0,1)

# plt.show()
axins = zoomed_inset_axes(ax, 2.5, loc='center right') # zoom-factor: 2.5, location: upper-left

axins.plot(all_depth[5:], stereo_err[5:] / all_depth_cnt_l[5:])
axins.plot(all_depth[5:], mono_err[5:] / all_depth_cnt_l[5:])


#
x1, x2, y1, y2 = 20, 300, 0, 0.2 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
# axins.set_yscale('log')

# plt.yticks(visible=False)
# plt.xticks(visible=False)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# prepare the demo image
# Z, extent = get_demo_image()
# Z2 = np.zeros([150, 150], dtype="d")
# ny, nx = Z.shape
# Z2[30:30+ny, 30:30+nx] = Z
#
# # extent = [-3, 4, -4, 3]
# ax.imshow(Z2, extent=extent, interpolation="nearest",
#           origin="lower")
#
# axins = zoomed_inset_axes(ax, 6, loc=1) # zoom = 6
# axins.imshow(Z2, extent=extent, interpolation="nearest",
#              origin="lower")
# #
# # # sub region of the original image
# x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
#
# plt.xticks(visible=False)
# plt.yticks(visible=False)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
ax.set_title('Relative L1 Error for Depth')
ax.set_xlabel('Depth')
ax.set_ylabel('Error')
plt.draw()
plt.show()