# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.utils.data as data
import os, math
import sys
import time
import random
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import datasets
import pcl

from fcn.config import cfg
from utils.blob import chromatic_transform, add_noise
from utils import mask as util_


class OSDObject(data.Dataset, datasets.imdb):
    def __init__(self, image_set, osd_object_path = None):

        self._name = 'osd_object_' + image_set
        self._image_set = image_set
        self._osd_object_path = self._get_default_path() if osd_object_path is None \
                            else osd_object_path
        self._classes_all = ('__background__', 'foreground')
        self._classes = self._classes_all
        self._pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
        self._width = 640
        self._height = 480

        # get all images
        data_path = os.path.join(self._osd_object_path, 'image_color')
        self.image_files = sorted(glob.glob(data_path + '/*.png'))

        print('%d images for dataset %s' % (len(self.image_files), self._name))
        self._size = len(self.image_files)
        assert os.path.exists(self._osd_object_path), \
                'osd_object path does not exist: {}'.format(self._osd_object_path)


    def process_label(self, foreground_labels):
        """ Process foreground_labels
                - Map the foreground_labels to {0, 1, ..., K-1}

            @param foreground_labels: a [H x W] numpy array of labels

            @return: foreground_labels
        """
        # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(foreground_labels)
        mapped_labels = foreground_labels.copy()
        for k in range(unique_nonnegative_indices.shape[0]):
            mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
        foreground_labels = mapped_labels
        return foreground_labels


    def __getitem__(self, idx):

        # BGR image
        filename = self.image_files[idx]
        im = cv2.imread(filename)
        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = chromatic_transform(im)
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = add_noise(im)
        im_tensor = torch.from_numpy(im) / 255.0

        im_tensor_bgr = im_tensor.clone()
        im_tensor_bgr = im_tensor_bgr.permute(2, 0, 1)

        im_tensor -= self._pixel_mean
        image_blob = im_tensor.permute(2, 0, 1)

        # Label
        labels_filename = filename.replace('image_color', 'annotation')
        foreground_labels = util_.imread_indexed(labels_filename)
        foreground_labels = self.process_label(foreground_labels)
        label_blob = torch.from_numpy(foreground_labels).unsqueeze(0)

        index = filename.find('OSD')
        sample = {'image_color': image_blob,
                  'image_color_bgr': im_tensor_bgr,
                  'label': label_blob,
                  'filename': filename[index+4:]}

        # Depth image
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            pcd_filename = filename.replace('image_color', 'pcd')
            pcd_filename = pcd_filename.replace('png', 'pcd')
            pcloud = pcl.load(pcd_filename).to_array()
            pcloud[np.isnan(pcloud)] = 0
            xyz_img = pcloud.reshape((self._height, self._width, 3))
            depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
            sample['depth'] = depth_blob

        return sample


    def __len__(self):
        return self._size


    def _get_default_path(self):
        """
        Return the default path where osd_object is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'OSD')
