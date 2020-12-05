#!/usr/bin/env python3

# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a PoseCNN on images"""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import cv2
import scipy.io
import glob

import _init_paths
from fcn.test_imageset import test_image_segmentation
from fcn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_dataset
import networks
from utils.blob import pad_im
from fcn.segmentation import Segmentor


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a PoseCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--pretrained_rrn', dest='pretrained_rrn',
                        help='initialize with pretrained checkpoint RRN',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--depth', dest='depth_name',
                        help='depth image pattern',
                        default='*depth.png', type=str)
    parser.add_argument('--color', dest='color_name',
                        help='color image pattern',
                        default='*color.png', type=str)
    parser.add_argument('--imgdir', dest='imgdir',
                        help='path of the directory with the test images',
                        default=None, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)
    parser.add_argument('--image_path', dest='image_path',
                        help='path to images', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


# save data
def save_data(file_rgb, out_label_refined, roi, features_crop):

    # meta data
    '''
    meta = {'roi': roi, 'features': features_crop.cpu().detach().numpy(), 'labels': out_label_refined.cpu().detach().numpy()}
    filename = file_rgb[:-9] + 'meta.mat'
    scipy.io.savemat(filename, meta, do_compression=True)
    print('save data to {}'.format(filename))
    '''

    # segmentation labels
    label_save = out_label_refined.cpu().detach().numpy()[0]
    label_save = np.clip(label_save, 0, 1) * 255
    label_save = label_save.astype(np.uint8)
    filename = file_rgb[:-4] + '-label.png'
    cv2.imwrite(filename, label_save)
    print('save data to {}'.format(filename))


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if len(cfg.TEST.CLASSES) == 0:
        cfg.TEST.CLASSES = cfg.TRAIN.CLASSES
    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.gpu_id = 0
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    cfg.instance_id = 0
    print('GPU device {:d}'.format(args.gpu_id))

    # list images
    if args.imgdir is not None:
        images_color = []
        filename = os.path.join(args.imgdir, args.color_name)
        files = glob.glob(filename)
        for i in range(len(files)):
            filename = files[i]
            images_color.append(filename)
        images_color.sort()
    elif args.image_path is not None:
        images_color = args.image_path.split(' ')

    # dataset
    cfg.MODE = 'TEST'
    dataset = get_dataset(args.dataset_name)

    # overwrite intrinsics
    if len(cfg.INTRINSICS) > 0:
        K = np.array(cfg.INTRINSICS).reshape(3, 3)
        if cfg.TEST.SCALES_BASE[0] != 1:
            scale = cfg.TEST.SCALES_BASE[0]
            K[0, 0] *= scale
            K[0, 2] *= scale
            K[1, 1] *= scale
            K[1, 2] *= scale
        dataset._intrinsic_matrix = K
        print(dataset._intrinsic_matrix)

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()

    network = networks.__dict__[args.network_name](dataset.num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
    cudnn.benchmark = True
    network.eval()

    # prepre region refinement network
    if args.pretrained_rrn:
        params = {
            # Padding for Region Refinement Network
            'padding_percentage' : 0.25,
            # Open/Close Morphology for IMP (Initial Mask Processing) module
            'use_open_close_morphology' : True,
            'open_close_morphology_ksize' : 9,
            # Closest Connected Component for IMP module
            'use_closest_connected_component' : True,
        }
        segmentor = Segmentor(params, args.pretrained_rrn)
    else:
        segmentor = None

    if cfg.TEST.VISUALIZE:
        index_images = np.random.permutation(len(images_color))
    else:
        index_images = range(len(images_color))

    for i in index_images:
        if os.path.exists(images_color[i]):
            # read images
            img = pad_im(cv2.imread(images_color[i], cv2.IMREAD_COLOR), 16)
            print(images_color[i])

            out_label, out_label_refined, roi = test_image_segmentation(i, network, dataset, img, segmentor)

            # save result
            if not cfg.TEST.VISUALIZE:
                save_data(images_color[i], out_label_refined, roi, features_crop)
        else:
            print('files not exist %s' % (images_color[i]))
