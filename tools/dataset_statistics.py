# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import os
import os.path
import torch
import cv2
import numpy as np
import glob
import random
import math
from transforms3d.quaternions import mat2quat, quat2mat
import _init_paths
from datasets import OCIDObject, OSDObject
import matplotlib.pyplot as plt
from utils import mask as util_

if __name__ == '__main__':
    dataset = OSDObject('test')
    num = dataset._size
    num_objects = []
    for i in range(num):

        filename = str(dataset.image_files[i])
        # labels_filename = filename.replace('rgb', 'label')
        labels_filename = filename.replace('image_color', 'annotation')

        foreground_labels = util_.imread_indexed(labels_filename)
        # mask table as background
        foreground_labels[foreground_labels == 1] = 0
        if 'table' in labels_filename:
            foreground_labels[foreground_labels == 2] = 0
        foreground_labels = dataset.process_label(foreground_labels)
        n = len(np.unique(foreground_labels)) - 1
        num_objects.append(n)
        print(labels_filename, n)

    nums = np.array(num_objects)
    print('min: %d' % (np.min(nums)))
    print('max: %d' % (np.max(nums)))
    print('mean: %f' % (np.mean(nums)))
