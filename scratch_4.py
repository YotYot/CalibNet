import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
import random
import pickle

pickle_file = '/home/yotamg/PycharmProjects/DL_project/surface.pickle'

with open(pickle_file, 'rb') as f:
    [ANGLES, TILTS, l] = pickle.load(f, encoding='latin1')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# x = y = np.arange(-3.0, 3.0, 0.05)
# X, Y = np.meshgrid(x, y)
# zs = np.array(fun(np.ravel(X), np.ravel(Y)))
# Z = zs.reshape(X.shape)


ax.plot_surface(TILTS ,ANGLES, l, cmap='jet')
ax.set_xlim(4,0)
ax.set_ylabel('Image Plane Rotation (Deg.)')
ax.set_xlabel('X Perspective Rotation (Deg.)')
ax.set_zlabel('L1 Difference Mono / Stereo')
ax.set_title("Mono / Stereo L1 Difference for Rotation")

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# x = y = np.arange(-3.0, 3.0, 0.05)
# X, Y = np.meshgrid(x, y)
# zs = np.array(fun(np.ravel(X), np.ravel(Y)))
# Z = zs.reshape(X.shape)


ax.plot_surface(ANGLES, TILTS, l, cmap='copper')
ax.set_xlim(4,0)
ax.set_xlabel('Image Plane Rotation (Deg.)')
ax.set_ylabel('X Perspective Rotation (Deg.)')
ax.set_zlabel('L1 Difference Mono / Stereo')
ax.set_title("Mono / Stereo L1 Difference for Rotation")

plt.show()