#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test a PoseCNN on images"""

from typing import TypedDict

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn as nn

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import cv2
import scipy.io
import glob
import json

# import tools._init_paths as _init_paths
# from lib.fcn.test_dataset import test_sample
# from lib.fcn.config import cfg, cfg_from_file, get_output_dir
# import lib.networks as networks
# from lib.utils.blob import pad_im
# from lib.utils import mask as util_
import tools._init_paths
from fcn.test_dataset import test_sample
from fcn.config import cfg, cfg_from_file, get_output_dir
import networks
from utils.blob import pad_im
from utils import mask as util_

RGB_Image_np = np.ndarray
DEPTH_Image_np = np.ndarray
RGB_Image = torch.Tensor
DEPTH_Image = torch.Tensor
VertexMap = torch.Tensor
class RGB_and_VertexMap(TypedDict):
    image_color: RGB_Image
    depth: VertexMap

class UnseenObjectClustering(nn.Module):

    def __init__(
        self,
        gpu_id: int = 0, #GPU id to use
        pretrained: str = "data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth", #Initialize with pretrained checkpoint
        pretrained_crop: str = "data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth", #Initialize with pretrained checkpoint for crops
        camera_params_filename: str = None, #Directory of where the camera-parameters json is
        cfg_file: str = "experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml", #Optional config file
        # dataset_name: str = 'shapenet_scene_train', #Dataset to train on
        randomize: bool = False, #Whether or not to randomize (do not use a fixed seed)
        network_name = 'seg_resnet34_8s_embedding', #Network
        # image_path = None, ################,
        visualize: bool = False, #Whether to visualise the segmentations or not
    ) -> None:

        super(UnseenObjectClustering, self).__init__()

        self.cfg = cfg
        if cfg_file is not None:
            self.cfg.NANA = "yess"
            cfg_from_file(cfg_file)

        if len(self.cfg.TEST.CLASSES) == 0:
            self.cfg.TEST.CLASSES = self.cfg.TRAIN.CLASSES
        self.cfg.TRAIN.EMBEDDING_METRIC = 'cosine'
        # print('Using config:')
        # pprint.pprint(cfg)

        if not randomize:
            # fix the random seeds (numpy and caffe) for reproducibility
            np.random.seed(cfg.RNG_SEED)

        # device
        self.cfg.gpu_id = 0
        self.cfg.device = torch.device('cuda:{:d}'.format(self.cfg.gpu_id))
        self.cfg.instance_id = 0
        num_classes = 2
        self.cfg.MODE = 'TEST'
        # cfg.TEST.VISUALIZE = visualize
        print('GPU device {:d}'.format(gpu_id))

        # check if intrinsics available
        if os.path.exists(camera_params_filename):
            with open(camera_params_filename) as f:
                self.camera_params = json.load(f)
        else:
            self.camera_params = None
            raise Warning("The camera parameters were not given")

        # prepare network
        if pretrained:
            network_data = torch.load(pretrained)
            # print("=> using pre-trained network '{}'".format(pretrained))
        else:
            network_data = None
            print("no pretrained network specified")
            sys.exit()

        self.network = networks.__dict__[network_name](num_classes, self.cfg.TRAIN.NUM_UNITS, network_data).cuda(device=self.cfg.device)
        self.network = torch.nn.DataParallel(self.network, device_ids=[self.cfg.gpu_id]).cuda(device=self.cfg.device)
        cudnn.benchmark = True
        self.network.eval()

        if pretrained_crop:
            network_data_crop = torch.load(pretrained_crop)
            self.network_crop = networks.__dict__[network_name](num_classes, self.cfg.TRAIN.NUM_UNITS, network_data_crop).cuda(device=self.cfg.device)
            self.network_crop = torch.nn.DataParallel(self.network_crop, device_ids=[self.cfg.gpu_id]).cuda(device=self.cfg.device)
            self.network_crop.eval()
        else:
            self.network_crop = None

        # if cfg.TEST.VISUALIZE:
        #     index_images = np.random.permutation(len(images_color))
        # else:
        #     index_images = range(len(images_color))

    def compute_xyz(self, depth_img, fx, fy, px, py, height, width) -> VertexMap:
        indices = util_.build_matrix_of_indices(height, width)
        z_e = depth_img
        x_e = (indices[..., 1] - px) * z_e / fx
        y_e = (indices[..., 0] - py) * z_e / fy
        xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
        return xyz_img

    def read_sample(self, rgb: RGB_Image_np, depth: DEPTH_Image_np, camera_params) -> RGB_and_VertexMap:
        # bgr image
        bgr_permute = [2, 1, 0]
        bgr = rgb[..., bgr_permute]

        if self.cfg.INPUT == 'DEPTH' or self.cfg.INPUT == 'RGBD':
            # depth image
            # depth_img = cv2.imread(filename_depth, cv2.IMREAD_ANYDEPTH)
            depth = depth.astype(np.float32) / 1000.0

            height = depth.shape[0]
            width = depth.shape[1]
            fx = camera_params['fx']
            fy = camera_params['fy']
            px = camera_params['x_offset']
            py = camera_params['y_offset']
            xyz_img = self.compute_xyz(depth, fx, fy, px, py, height, width)
        else:
            xyz_img = None

        im_tensor = torch.from_numpy(bgr) / 255.0
        pixel_mean = torch.tensor(self.cfg.PIXEL_MEANS / 255.0).float()
        im_tensor -= pixel_mean
        image_blob = im_tensor.permute(2, 0, 1)
        sample = {'image_color': image_blob.unsqueeze(0)}

        if self.cfg.INPUT == 'DEPTH' or self.cfg.INPUT == 'RGBD':
            depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
            sample['depth'] = depth_blob.unsqueeze(0)

        return sample

    def forward(
        self,
        rgb_img: RGB_Image_np,
        depth_img: DEPTH_Image_np,
    ):
        assert rgb_img.shape[-1] == 3, "The rgb image must have the 3 channels at the end"
        assert rgb_img.shape[:-1] == depth_img.shape, f"The rgb and depth images have different shapes, {rgb_img.shape} and {depth_img.shape} respectively"

        single_image = False
        if len(rgb_img.shape) == 3:
            rgb_img = np.expand_dims(rgb_img, axis=0)
            depth_img = np.expand_dims(depth_img, axis=0)
            single_image = True

        n_images = rgb_img.shape[0]
        predictions = []

        for i in range(n_images):
            sample = self.read_sample(rgb_img[i], depth_img[i], self.camera_params)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            out_label, out_label_refined = test_sample(sample, self.network, self.network_crop)
            predictions.append(out_label_refined)

        if single_image:
            return predictions[0]
        return predictions
