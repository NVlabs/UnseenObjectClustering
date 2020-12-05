# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import os
import os.path as osp
import numpy as np
import datasets
import math
import glob
from fcn.config import cfg

class imdb(object):
    """Image database."""

    def __init__(self):
        self._name = ''
        self._num_classes = 0
        self._classes = []
        self._class_colors = []

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(datasets.ROOT_DIR, 'data', 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path


    # backproject pixels into 3D points in camera's coordinate system
    def backproject(self, depth_cv, intrinsic_matrix, factor):

        depth = depth_cv.astype(np.float32, copy=True) / factor

        index = np.where(~np.isfinite(depth))
        depth[index[0], index[1]] = 0

        # get intrinsic matrix
        K = intrinsic_matrix
        Kinv = np.linalg.inv(K)

        # compute the 3D points
        width = depth.shape[1]
        height = depth.shape[0]

        # construct the 2D points matrix
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

        # backprojection
        R = np.dot(Kinv, x2d.transpose())

        # compute the 3D points
        X = np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)
        return np.array(X).transpose().reshape((height, width, 3))


    def _build_uniform_poses(self):

        self.eulers = []
        interval = cfg.TRAIN.UNIFORM_POSE_INTERVAL
        for yaw in range(-180, 180, interval):
            for pitch in range(-90, 90, interval):
                for roll in range(-180, 180, interval):
                    self.eulers.append([yaw, pitch, roll])

        # sample indexes
        num_poses = len(self.eulers)
        num_classes = len(self._classes_all) - 1 # no background
        self.pose_indexes = np.zeros((num_classes, ), dtype=np.int32)
        self.pose_lists = []
        for i in range(num_classes):
            self.pose_lists.append(np.random.permutation(np.arange(num_poses)))


    def _build_background_images(self):

        backgrounds_color = []
        backgrounds_depth = []
        if cfg.TRAIN.SYN_BACKGROUND_SPECIFIC:
            # NVIDIA
            '''
            allencenter = os.path.join(self.cache_path, '../AllenCenter/data')
            subdirs = os.listdir(allencenter)
            for i in xrange(len(subdirs)):
                subdir = subdirs[i]
                files = os.listdir(os.path.join(allencenter, subdir))
                for j in range(len(files)):
                    filename = os.path.join(allencenter, subdir, files[j])
                    backgrounds_color.append(filename)
            '''

            comotion = os.path.join(self.cache_path, '../D435-data-with-depth/data')
            subdirs = os.listdir(comotion)
            for i in xrange(len(subdirs)):
                subdir = subdirs[i]
                files = os.listdir(os.path.join(comotion, subdir))
                for j in range(len(files)):
                    filename = os.path.join(comotion, subdir, files[j])
                    if 'depth.png' in filename:
                        backgrounds_depth.append(filename)
                    else:
                        backgrounds_color.append(filename)

            backgrounds_color.sort()
            backgrounds_depth.sort()
        else:
            '''
            # SUN 2012
            root = os.path.join(self.cache_path, '../SUN2012/data/Images')
            subdirs = os.listdir(root)

            for i in xrange(len(subdirs)):
                subdir = subdirs[i]
                names = os.listdir(os.path.join(root, subdir))

                for j in xrange(len(names)):
                    name = names[j]
                    if os.path.isdir(os.path.join(root, subdir, name)):
                        files = os.listdir(os.path.join(root, subdir, name))
                        for k in range(len(files)):
                            if os.path.isdir(os.path.join(root, subdir, name, files[k])):
                                filenames = os.listdir(os.path.join(root, subdir, name, files[k]))
                                for l in range(len(filenames)):
                                    filename = os.path.join(root, subdir, name, files[k], filenames[l])
                                    backgrounds.append(filename)
                            else:
                                filename = os.path.join(root, subdir, name, files[k])
                                backgrounds.append(filename)
                    else:
                        filename = os.path.join(root, subdir, name)
                        backgrounds.append(filename)

            # ObjectNet3D
            objectnet3d = os.path.join(self.cache_path, '../ObjectNet3D/data')
            files = os.listdir(objectnet3d)
            for i in range(len(files)):
                filename = os.path.join(objectnet3d, files[i])
                backgrounds.append(filename)
            '''

            # PASCAL 2012
            pascal = os.path.join(self.cache_path, '../PASCAL2012/data')
            files = os.listdir(pascal)
            for i in range(len(files)):
                filename = os.path.join(pascal, files[i])
                backgrounds_color.append(filename)

            '''
            # YCB Background
            ycb = os.path.join(self.cache_path, '../YCB_Background')
            files = os.listdir(ycb)
            for i in range(len(files)):
                filename = os.path.join(ycb, files[i])
                backgrounds.append(filename)
            '''

        # depth background
        kinect = os.path.join(self.cache_path, '../Kinect')
        subdirs = os.listdir(kinect)
        for i in xrange(len(subdirs)):
            subdir = subdirs[i]
            files = glob.glob(os.path.join(self.cache_path, '../Kinect', subdir, '*depth*'))
            for j in range(len(files)):
                filename = os.path.join(self.cache_path, '../Kinect', subdir, files[j])
                backgrounds_depth.append(filename)

        for i in xrange(len(backgrounds_color)):
            if not os.path.isfile(backgrounds_color[i]):
                print('file not exist {}'.format(backgrounds_color[i]))

        for i in xrange(len(backgrounds_depth)):
            if not os.path.isfile(backgrounds_depth[i]):
                print('file not exist {}'.format(backgrounds_depth[i]))

        self._backgrounds_color = backgrounds_color
        self._backgrounds_depth = backgrounds_depth
        print('build color background images finished, {:d} images'.format(len(backgrounds_color)))
        print('build depth background images finished, {:d} images'.format(len(backgrounds_depth)))
