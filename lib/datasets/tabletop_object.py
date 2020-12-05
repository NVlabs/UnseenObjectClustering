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
import numpy.random as npr
import cv2
import glob
import matplotlib.pyplot as plt
import datasets

from fcn.config import cfg
from utils.blob import chromatic_transform, add_noise
from utils import augmentation
from utils import mask as util_

data_loading_params = {
    
    # Camera/Frustum parameters
    'img_width' : 640, 
    'img_height' : 480,
    'near' : 0.01,
    'far' : 100,
    'fov' : 45, # vertical field of view in degrees
    
    'use_data_augmentation' : True,

    # Multiplicative noise
    'gamma_shape' : 1000.,
    'gamma_scale' : 0.001,
    
    # Additive noise
    'gaussian_scale' : 0.005, # 5mm standard dev
    'gp_rescale_factor' : 4,
    
    # Random ellipse dropout
    'ellipse_dropout_mean' : 10, 
    'ellipse_gamma_shape' : 5.0, 
    'ellipse_gamma_scale' : 1.0,

    # Random high gradient dropout
    'gradient_dropout_left_mean' : 15, 
    'gradient_dropout_alpha' : 2., 
    'gradient_dropout_beta' : 5.,

    # Random pixel dropout
    'pixel_dropout_alpha' : 1., 
    'pixel_dropout_beta' : 10.,    
}

def compute_xyz(depth_img, camera_params):
    """ Compute ordered point cloud from depth image and camera parameters.

        If focal lengths fx,fy are stored in the camera_params dictionary, use that.
        Else, assume camera_params contains parameters used to generate synthetic data (e.g. fov, near, far, etc)

        @param depth_img: a [H x W] numpy array of depth values in meters
        @param camera_params: a dictionary with parameters of the camera used 
    """

    # Compute focal length from camera parameters
    if 'fx' in camera_params and 'fy' in camera_params:
        fx = camera_params['fx']
        fy = camera_params['fy']
    else: # simulated data
        aspect_ratio = camera_params['img_width'] / camera_params['img_height']
        e = 1 / (np.tan(np.radians(camera_params['fov']/2.)))
        t = camera_params['near'] / e; b = -t
        r = t * aspect_ratio; l = -r
        alpha = camera_params['img_width'] / (r-l) # pixels per meter
        focal_length = camera_params['near'] * alpha # focal length of virtual camera (frustum camera)
        fx = focal_length; fy = focal_length

    if 'x_offset' in camera_params and 'y_offset' in camera_params:
        x_offset = camera_params['x_offset']
        y_offset = camera_params['y_offset']
    else: # simulated data
        x_offset = camera_params['img_width']/2
        y_offset = camera_params['img_height']/2

    indices = util_.build_matrix_of_indices(camera_params['img_height'], camera_params['img_width'])
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    
    return xyz_img


class TableTopObject(data.Dataset, datasets.imdb):
    def __init__(self, image_set, tabletop_object_path = None):

        self._name = 'tabletop_object_' + image_set
        self._image_set = image_set
        self._tabletop_object_path = self._get_default_path() if tabletop_object_path is None \
                            else tabletop_object_path
        self._classes_all = ('__background__', 'foreground')
        self._classes = self._classes_all
        self._pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
        self.params = data_loading_params

        # crop dose not use background
        if cfg.TRAIN.SYN_CROP:
            self.NUM_VIEWS_PER_SCENE = 5
        else:
            self.NUM_VIEWS_PER_SCENE = 7

        # get a list of all scenes
        if image_set == 'train':
            data_path = os.path.join(self._tabletop_object_path, 'training_set')
            self.scene_dirs = sorted(glob.glob(data_path + '/*'))
        elif image_set == 'test':
            data_path = os.path.join(self._tabletop_object_path, 'test_set')
            print(data_path)
            self.scene_dirs = sorted(glob.glob(data_path + '/*'))
        elif image_set == 'all':
            data_path = os.path.join(self._tabletop_object_path, 'training_set')
            scene_dirs_train = sorted(glob.glob(data_path + '/*'))
            data_path = os.path.join(self._tabletop_object_path, 'test_set')
            scene_dirs_test = sorted(glob.glob(data_path + '/*'))
            self.scene_dirs = scene_dirs_train + scene_dirs_test

        print('%d scenes for dataset %s' % (len(self.scene_dirs), self._name))
        self._size = len(self.scene_dirs) * self.NUM_VIEWS_PER_SCENE
        assert os.path.exists(self._tabletop_object_path), \
                'tabletop_object path does not exist: {}'.format(self._tabletop_object_path)


    def process_depth(self, depth_img):
        """ Process depth channel
                - change from millimeters to meters
                - cast to float32 data type
                - add random noise
                - compute xyz ordered point cloud
        """

        # millimeters -> meters
        depth_img = (depth_img / 1000.).astype(np.float32)

        # add random noise to depth
        if self.params['use_data_augmentation']:
            depth_img = augmentation.add_noise_to_depth(depth_img, self.params)
            depth_img = augmentation.dropout_random_ellipses(depth_img, self.params)

        # Compute xyz ordered point cloud and add noise
        xyz_img = compute_xyz(depth_img, self.params)
        if self.params['use_data_augmentation']:
            xyz_img = augmentation.add_noise_to_xyz(xyz_img, depth_img, self.params)
        return xyz_img


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


    def pad_crop_resize(self, img, label, depth):
        """ Crop the image around the label mask, then resize to 224x224
        """

        H, W, _ = img.shape

        # sample an object to crop
        K = np.max(label)
        while True:
            if K > 0:
                idx = np.random.randint(1, K+1)
            else:
                idx = 0
            foreground = (label == idx).astype(np.float32)

            # get tight box around label/morphed label
            x_min, y_min, x_max, y_max = util_.mask_to_tight_box(foreground)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2

            # make bbox square
            x_delta = x_max - x_min
            y_delta = y_max - y_min
            if x_delta > y_delta:
                y_min = cy - x_delta / 2
                y_max = cy + x_delta / 2
            else:
                x_min = cx - y_delta / 2
                x_max = cx + y_delta / 2

            sidelength = x_max - x_min
            padding_percentage = np.random.uniform(cfg.TRAIN.min_padding_percentage, cfg.TRAIN.max_padding_percentage)
            padding = int(round(sidelength * padding_percentage))
            if padding == 0:
                padding = 25

            # Pad and be careful of boundaries
            x_min = max(int(x_min - padding), 0)
            x_max = min(int(x_max + padding), W-1)
            y_min = max(int(y_min - padding), 0)
            y_max = min(int(y_max + padding), H-1)

            # crop
            if (y_min == y_max) or (x_min == x_max):
                continue

            img_crop = img[y_min:y_max+1, x_min:x_max+1]
            label_crop = label[y_min:y_max+1, x_min:x_max+1]
            roi = [x_min, y_min, x_max, y_max]
            if depth is not None:
                depth_crop = depth[y_min:y_max+1, x_min:x_max+1]
            break

        # resize
        s = cfg.TRAIN.SYN_CROP_SIZE
        img_crop = cv2.resize(img_crop, (s, s))
        label_crop = cv2.resize(label_crop, (s, s), interpolation=cv2.INTER_NEAREST)
        if depth is not None:
            depth_crop = cv2.resize(depth_crop, (s, s), interpolation=cv2.INTER_NEAREST)
        else:
            depth_crop = None

        return img_crop, label_crop, depth_crop


    # sample num of pixel for clustering instead of using all
    def sample_pixels(self, labels, num=1000):
        # -1 ignore
        labels_new = -1 * np.ones_like(labels)
        K = np.max(labels)
        for i in range(K+1):
            index = np.where(labels == i)
            n = len(index[0])
            if n <= num:
                labels_new[index[0], index[1]] = i
            else:
                perm = np.random.permutation(n)
                selected = perm[:num]
                labels_new[index[0][selected], index[1][selected]] = i
        return labels_new


    def __getitem__(self, idx):

        # Get scene directory, crop dose not use background
        scene_idx = idx // self.NUM_VIEWS_PER_SCENE
        scene_dir = self.scene_dirs[scene_idx]

        # Get view number
        view_num = idx % self.NUM_VIEWS_PER_SCENE
        if cfg.TRAIN.SYN_CROP:
            view_num += 2

        # Label
        foreground_labels_filename = os.path.join(scene_dir, 'segmentation_%05d.png' % view_num)
        foreground_labels = util_.imread_indexed(foreground_labels_filename)
        # mask table as background
        foreground_labels[foreground_labels == 1] = 0
        foreground_labels = self.process_label(foreground_labels)

        # BGR image
        filename = os.path.join(scene_dir, 'rgb_%05d.jpeg' % view_num)
        im = cv2.imread(filename)

        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            # Depth image
            depth_img_filename = os.path.join(scene_dir, 'depth_%05d.png' % view_num)
            depth_img = cv2.imread(depth_img_filename, cv2.IMREAD_ANYDEPTH) # This reads a 16-bit single-channel image. Shape: [H x W]
            xyz_img = self.process_depth(depth_img)
        else:
            xyz_img = None

        # crop
        if cfg.TRAIN.SYN_CROP:
            im, foreground_labels, xyz_img = self.pad_crop_resize(im, foreground_labels, xyz_img)
            foreground_labels = self.process_label(foreground_labels)

        # sample labels
        if cfg.TRAIN.EMBEDDING_SAMPLING:
            foreground_labels = self.sample_pixels(foreground_labels, cfg.TRAIN.EMBEDDING_SAMPLING_NUM)

        label_blob = torch.from_numpy(foreground_labels).unsqueeze(0)
        sample = {'label': label_blob}

        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = chromatic_transform(im)
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = add_noise(im)
        im_tensor = torch.from_numpy(im) / 255.0
        im_tensor -= self._pixel_mean
        image_blob = im_tensor.permute(2, 0, 1)
        sample['image_color'] = image_blob

        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
            sample['depth'] = depth_blob

        return sample


    def __len__(self):
        return self._size


    def _get_default_path(self):
        """
        Return the default path where tabletop_object is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'tabletop')
