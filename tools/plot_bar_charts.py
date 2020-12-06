# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 4

# F1_maskrcnn = (62.7, 84.7, 78.1, 76.6, 76.0)
# F1_ours = (59.4, 86.4, 82.8, 88.5, 82.9)

# F1_maskrcnn = (54.6, 78.8, 70.8, 64.3, 64.7)
# F1_ours = (36.5, 75.6, 67.2, 78.8, 68.5)

# F1_maskrcnn = (59.4, 86.4, 82.8, 88.5, 82.9)
# F1_ours = (58.1, 86.4, 84.0, 87.8, 85.1)

# F1_maskrcnn = (36.5, 75.6, 67.2, 78.8, 68.5)
# F1_ours = (40.8, 79.6, 72.5, 82.3, 78.3)


# F1_overlap = (84.7, 81.7, 86.4, 87.8)
# F1_boundary = (78.8, 71.4, 76.2, 82.3)
# percentage = (72.7, 69.1, 77.2, 85.6)

F1_overlap = (80.6, 79.9, 83.3, 87.4)
F1_boundary = (54.6, 65.6, 71.2, 69.4)
percentage = (77.6, 71.9, 73.8, 83.2)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8

rects1 = plt.bar(index, F1_overlap, bar_width,
alpha=opacity,
color='b',
label='F1_overlap')

rects2 = plt.bar(index + bar_width, F1_boundary, bar_width,
alpha=opacity,
color='g',
label='F1_boundary')

rects3 = plt.bar(index + 2 * bar_width, percentage, bar_width,
alpha=opacity,
color='r',
label='%75')

plt.xlabel('Methods')
# plt.ylabel('F1 boundary')
plt.title('OSD (111 images)')
plt.xticks(index + bar_width, ('MRCNN Depth', 'UOIS-2D', 'UOIS-3D', 'Ours'))
plt.legend(loc='lower left')

labels = F1_overlap
for i, v in enumerate(labels):
    ax.text(i-.2, v+1, 
              labels[i], 
              fontsize=12, 
              color='k')

labels = F1_boundary
for i, v in enumerate(labels):
    ax.text(i+.1, v+1, 
              labels[i], 
              fontsize=12, 
              color='k')

labels = percentage
for i, v in enumerate(labels):
    ax.text(i+.35, v+1, 
              labels[i], 
              fontsize=12, 
              color='k')

plt.tight_layout()
plt.show()
