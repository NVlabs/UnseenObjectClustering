# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

t = np.arange(start=1, stop=17, step=1)
rc('font', weight='bold')

# UCN RGB

F1_overlap_rgb = (0.206197, 0.357676, 0.345578, 0.455547, 0.457957, 0.502031, 0.457381, 0.518946, 0.552038, 0.520665, \
              0.50691, 0.519003, 0.550123, 0.514964, 0.514002, 0.593742)
F1_boundary_rgb = (0.094635, 0.167361, 0.172439, 0.237247, 0.244162, 0.270599, 0.258174, 0.309146, 0.327651, 0.316562, \
               0.322138, 0.311927, 0.344229, 0.308593, 0.329168, 0.364516)
percentage_rgb = (0.263843, 0.355614, 0.356287, 0.414091, 0.405764, 0.431488, 0.435723, 0.464883, 0.471861, 0.475419, \
              0.503157, 0.497285, 0.506068, 0.473288, 0.497579, 0.480164)

# UCN Depth

F1_overlap_depth = (0.632557, 0.745917, 0.775232, 0.802915, 0.82635, 0.834976, 0.843941, 0.836614, 0.857734, 0.858773, \
                    0.846244, 0.853272, 0.843275, 0.8384, 0.846614, 0.864338)

F1_boundary_depth = (0.219215, 0.327336, 0.414885, 0.471119, 0.590424, 0.615502, 0.668548, 0.656816, 0.714789, 0.726485, \
                     0.721683, 0.717682, 0.723354, 0.724921, 0.738028, 0.756031)

percentage_depth = (0.463543, 0.572034, 0.607148, 0.654096, 0.700107, 0.700688, 0.72621, 0.719467, 0.76059, 0.751082, \
                    0.733714, 0.735936, 0.712744, 0.723239, 0.726254, 0.753693)

# UCN RGBD early

F1_overlap_rgbd_early = (0.357674, 0.553803, 0.607327, 0.661596, 0.707028, 0.721938, 0.741733, 0.77255, 0.795557, 0.735402, \
                         0.806955, 0.758339, 0.800102, 0.815694, 0.799456, 0.828135)

F1_boundary_rgbd_early = (0.128438, 0.281023, 0.362007, 0.432142, 0.481427, 0.476286, 0.510337, 0.559285, 0.595986, 0.535778, \
                          0.621609, 0.593379, 0.59994, 0.646276, 0.637706, 0.672144)

percentage_rgbd_early = (0.290032, 0.420344, 0.497644, 0.555368, 0.597204, 0.576219, 0.598361, 0.665128, 0.687534, 0.635226, \
                         0.683646, 0.670646, 0.677623, 0.698645, 0.716388, 0.735246)

# UCN RGBD add

F1_overlap_rgbd_add = (0.514279, 0.662002, 0.795837, 0.788407, 0.795113, 0.842289, 0.824394, 0.854453, 0.847598, 0.865754, \
                       0.855248, 0.85502, 0.857568, 0.856234, 0.840809, 0.884881)

F1_boundary_rgbd_add = (0.245276, 0.324417, 0.549822, 0.534663, 0.576119, 0.679746, 0.639074, 0.705335, 0.722362, 0.742819, \
                        0.749845, 0.73857, 0.758677, 0.755076, 0.739145, 0.787763)

percentage_rgbd_add = (0.491431, 0.538068, 0.661125, 0.675489, 0.695592, 0.742781, 0.731744, 0.744917, 0.736696, 0.766834, \
                       0.747862, 0.741274, 0.76629, 0.747441, 0.723242, 0.821638)

# UCN RGBD cat

F1_overlap_rgbd_cat = (0.441337, 0.591691, 0.747262, 0.727342, 0.807502, 0.817291, 0.816996, 0.827194, 0.831351, 0.841048, \
                       0.808059, 0.834401, 0.835638, 0.835728, 0.806224, 0.828991)

F1_boundary_rgbd_cat = (0.190999, 0.286006, 0.397822, 0.452141, 0.567425, 0.576083, 0.598294, 0.645848, 0.670346, 0.682605, \
                        0.587685, 0.674055, 0.713088, 0.700418, 0.607698, 0.685053)

percentage_rgbd_cat = (0.475042, 0.531699, 0.617873, 0.639375, 0.673361, 0.678608, 0.677335, 0.701095, 0.705839, 0.709701, \
                       0.662733, 0.7124, 0.724381, 0.71867, 0.676644, 0.682604)

# create plot
size = 12
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(t, F1_overlap_rgb, marker='o', color='r')
plt.plot(t, F1_overlap_depth, marker='o', color='g')
plt.plot(t, F1_overlap_rgbd_early, marker='o', color='b')
plt.plot(t, F1_overlap_rgbd_add, marker='o', color='c')
plt.plot(t, F1_overlap_rgbd_cat, marker='o', color='y')
ax.set_title('F Overlap', fontsize=size, fontweight='bold')
plt.xticks(t, fontsize=size)
plt.yticks(fontsize=size)
plt.xlabel('epoch', fontsize=size, fontweight='bold')
ax.legend(['UCN RGB', 'UCN Depth', 'UCN RGBD early', 'UCN RGBD add', 'UCN RGBD concat'], fontsize=size)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(t, F1_boundary_rgb, marker='o', color='r')
plt.plot(t, F1_boundary_depth, marker='o', color='g')
plt.plot(t, F1_boundary_rgbd_early, marker='o', color='b')
plt.plot(t, F1_boundary_rgbd_add, marker='o', color='c')
plt.plot(t, F1_boundary_rgbd_cat, marker='o', color='y')
ax.set_title('F Boundary', fontsize=size, fontweight='bold')
plt.xticks(t, fontsize=size)
plt.yticks(fontsize=size)
plt.xlabel('epoch', fontsize=size, fontweight='bold')
ax.legend(['UCN RGB', 'UCN Depth', 'UCN RGBD early', 'UCN RGBD add', 'UCN RGBD concat'], fontsize=size)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(t, percentage_rgb, marker='o', color='r')
plt.plot(t, percentage_depth, marker='o', color='g')
plt.plot(t, percentage_rgbd_early, marker='o', color='b')
plt.plot(t, percentage_rgbd_add, marker='o', color='c')
plt.plot(t, percentage_rgbd_cat, marker='o', color='y')
ax.set_title('%75', fontsize=size, fontweight='bold')
plt.xticks(t, fontsize=size)
plt.yticks(fontsize=size)
plt.xlabel('epoch', fontsize=size, fontweight='bold')
ax.legend(['UCN RGB', 'UCN Depth', 'UCN RGBD early', 'UCN RGBD add', 'UCN RGBD concat'], fontsize=size)
plt.show()
